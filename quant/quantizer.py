from typing import Literal
import torch
from torch import Tensor
from transformers import PreTrainedModel
from copy import deepcopy

class AttnQuantizer:
    def __init__(
        self,
        qbits: float,
        kbits: float,
        vbits: float,
        sym: bool,
        gsize: int = -1,
        qdtype: Literal["int", "fp"] = "int",
        kdtype: Literal["int", "fp"] = "int",
        vdtype: Literal["int", "fp"] = "int",
        qdim: Literal["token", "channel"]="token",
        kdim: Literal["token", "channel"]="token",
        vdim: Literal["token", "channel"]="token",
        window: int = 0,
        method: Literal["rtn", "had", "sage"]="rtn",
    ):
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.qbits = qbits
        self.kbits = kbits
        self.vbits = vbits
        self.qdtype = qdtype
        self.kdtype = kdtype
        self.vdtype = vdtype
        
        self.sym = sym
        self.gsize = gsize
        self.window = window
        self.method = method
    
    @staticmethod
    def from_qstr(qstr: str):
        """
        e.g. q8_k4_v4_g-1_sym_rtn_ttc_w0; qf8_kf8_vf8_g-1_sym_had_ttt_w32;
        """
        splits = qstr.strip().split("_")
        assert len(splits) == 8
        
        qbits = float(splits[0][1:]) if "f" not in splits[0] else 8
        kbits = float(splits[1][1:]) if "f" not in splits[1] else 8
        vbits = float(splits[2][1:]) if "f" not in splits[2] else 8
        qdtype = "int" if "f" not in splits[0] else "fp"
        kdtype = "int" if "f" not in splits[1] else "fp"
        vdtype = "int" if "f" not in splits[2] else "fp"
        gsize = int(splits[3][1:])
        sym = splits[4] == "sym"
        method = splits[5]
        qdim, kdim, vdim = [x for x in map(lambda a: "token" if a=="t" else "channel", splits[6])]
        window = int(splits[7][1:])
        return AttnQuantizer(
            qbits=qbits,
            kbits=kbits,
            vbits=vbits,
            sym=sym,
            gsize=gsize,
            qdtype=qdtype,
            kdtype=kdtype,
            vdtype=vdtype,
            qdim=qdim,
            kdim=kdim,
            vdim=vdim,
            window=window,
            method=method,
        )

    def is_quant(self):
        return self.qbits != 16 or self.kbits!= 16 or self.vbits!= 16
        
    def fake_quant(self, t: Tensor, ty: Literal["q", "k", "v"]):
        """
        t: [bs, nh, seqlen, d]
        """
        assert ty in ["q", "k", "v"]
        assert len(t.shape) == 4
        
        rdim = self.qdim if ty == "q" else (self.kdim if ty == "k" else self.vdim)
        bits = self.qbits if ty == "q" else (self.kbits if ty == "k" else self.vbits)
        gsize = self.gsize if self.gsize > 0 else t.shape[-1]
        dtype = self.qdtype if ty == "q" else (self.kdtype if ty == "k" else self.vdtype)
        
        ori_ty = t.dtype
        if bits >= 16:
            return t, None, None
        
        if rdim == "channel": assert gsize != -1
        
        if rdim == "channel":
            reshape_t = t.transpose(-1, -2)
            ori_shape = reshape_t.shape
            reshape_t = reshape_t.reshape(-1, gsize)
        else:
            ori_shape = t.shape
            reshape_t = t.reshape(-1, gsize)

        if self.sym:
            dmax = torch.finfo(torch.float8_e4m3fn).max if dtype == "fp" else 2**(bits-1)-1
            scale = reshape_t.abs().amax(keepdim=True).div(dmax)
            if dtype == "fp":
                quant = reshape_t.div(scale).to(torch.float8_e4m3fn).to(ori_ty).mul(scale)
            else:
                quant = reshape_t.div(scale).round().mul(scale)
            zero = 0
        else:
            assert dtype != "fp"
            cmin, cmax = reshape_t.aminmax(keepdim=True)
            scale = (cmax - cmin) / (2 ** bits - 1)
            quant = reshape_t.sub(cmin).div(scale).round().clamp(0, 2**bits-1).mul(scale).add(cmin)
            zero = cmin
        
        if rdim == "channel":
            quant = quant.reshape(*ori_shape).transpose(-1, -2)
        else:
            quant = quant.reshape(*ori_shape)
        
        return quant, scale, zero
    
    @staticmethod
    def plug_into_model(quantizer: "AttnQuantizer", model: PreTrainedModel):
        for layer in model.layers:
            layer.self_attn.quantizer = deepcopy(quantizer)
    
    @staticmethod
    def remove_from_model(model: PreTrainedModel):
        for layer in model.layers:
            layer.self_attn.quantizer = None
        