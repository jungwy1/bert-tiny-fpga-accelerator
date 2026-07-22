"""
Generate docs/params_reference.md -- the complete index of every name stored in
quant_params.pt and requant_params.pt.

Written by reading the actual files, so it never drifts from what is exported.
Re-run after changing export_params.py or compile_requant.py.

Run (from python/):  python dump_params.py
"""

import torch

QUANT = "quant_params.pt"
REQUANT = "requant_params.pt"
OUT = "../docs/params_reference.md"


def fmt(v):
    if torch.is_tensor(v):
        return f"`{tuple(v.shape)}` {str(v.dtype).replace('torch.','')}"
    if isinstance(v, float):
        return f"{v:.4e}"
    return f"`{v}`" if isinstance(v, str) else str(v)


def main():
    q = torch.load(QUANT, weights_only=True)
    r = torch.load(REQUANT, weights_only=True)
    L = []
    add = L.append

    add("# 파라미터 이름 레퍼런스\n")
    add("> `python dump_params.py`로 재생성. 실제 `.pt` 파일에서 읽어 만들므로 항상 최신.\n")
    add(f"> 원본: `{QUANT}`, `{REQUANT}`\n")

    # ---------------------------------------------------------- quant_params
    add("\n---\n\n## `quant_params.pt`\n")
    add("calibration 산출물. activation scale과 양자화된 weight/bias/gamma/beta/table.\n")

    add(f"\n### `act` — activation scale ({len(q['act'])}개)\n")
    add("| 이름 | scale | 관찰 absmax | 출처 |")
    add("|------|-------|------------|------|")
    for n, d in q["act"].items():
        add(f"| `{n}` | {d['scale']:.4e} | {d['observed_absmax']:.3f} | {d['source']} |")

    add(f"\n### `op` — 파라미터를 가진 연산 ({len(q['op'])}개)\n")
    groups = [("matmul", [k for k in q["op"] if k.startswith("W_") or ".W_" in k]),
              ("LayerNorm", [k for k in q["op"] if ".ln" in k or k == "emb_ln"]),
              ("embedding", [k for k in q["op"] if k.startswith("emb_") and k != "emb_ln"])]
    for title, names in groups:
        if not names:
            continue
        add(f"\n**{title}** ({len(names)}개)\n")
        keys = list(q["op"][names[0]].keys())
        add("| 이름 | " + " | ".join(f"`{k}`" for k in keys) + " |")
        add("|------|" + "|".join(["------"] * len(keys)) + "|")
        for n in names:
            add(f"| `{n}` | " + " | ".join(fmt(q["op"][n][k]) for k in keys) + " |")

    # -------------------------------------------------------- requant_params
    add("\n---\n\n## `requant_params.pt`\n")
    add("compile 산출물. 하드웨어가 쓰는 정수 상수만 담는다 (런타임 float 없음).\n")

    add(f"\n### `requant` — 누산기 → 다음 INT8 ({len(r['requant'])}개)\n")
    add("| 이름 | kind | M | M_int | shift | sign | acc비트 | 곱비트 |")
    add("|------|------|---|-------|-------|------|--------|-------|")
    for n, d in r["requant"].items():
        add(f"| `{n}` | {d['kind']} | {d['M']:.4e} | {d['M_int']} | {d['shift']} | "
            f"{d['sign']} | {d['acc_bits']} | {d['prod_bits']} |")

    add(f"\n### `align` — 덧셈 전 scale 정렬 ({len(r['align'])}개)\n")
    add("| 이름 | M | M_int | shift | 정렬 대상 |")
    add("|------|---|-------|-------|----------|")
    for n, d in r["align"].items():
        add(f"| `{n}` | {d['M']:.4e} | {d['M_int']} | {d['shift']} | `{d['operand']}` |")

    add(f"\n### `nonlin` — 비선형 유닛 하드와이어 상수 ({len(r['nonlin'])}개)\n")
    for t in ("gelu", "softmax", "tanh", "layernorm"):
        names = [k for k, d in r["nonlin"].items() if d["type"] == t]
        if not names:
            continue
        keys = [k for k in r["nonlin"][names[0]] if k != "type"]
        add(f"\n**{t}**\n")
        add("| 이름 | " + " | ".join(f"`{k}`" for k in keys) + " |")
        add("|------|" + "|".join(["------"] * len(keys)) + "|")
        for n in names:
            add(f"| `{n}` | " + " | ".join(fmt(r["nonlin"][n][k]) for k in keys) + " |")

    add(f"\n### `fused` — 연산에 융합, 곱수 없음 ({len(r['fused'])}개)\n")
    add("| 이름 | scale | 비고 |")
    add("|------|-------|------|")
    for n, d in r["fused"].items():
        add(f"| `{n}` | {d['scale']:.4e} | {d['note']} |")

    add(f"\n### 기타\n\n- `K_layernorm` = `{r['K_layernorm']}`\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(L)} lines)")
    print(f"  act {len(q['act'])} | op {len(q['op'])} | requant {len(r['requant'])} "
          f"| align {len(r['align'])} | nonlin {len(r['nonlin'])} | fused {len(r['fused'])}")


if __name__ == "__main__":
    main()
