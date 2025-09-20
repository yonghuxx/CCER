from .scorer import BERTScorer

with open("bert_score/hyps_long.txt") as f:
    cands = [line.strip() for line in f]

with open("bert_score/refs_long.txt") as f:
    refs = [line.strip() for line in f]

scorer = BERTScorer(model_type="roberta-large",lang="en", rescale_with_baseline=True)
P, R, F1 = scorer.score(cands, refs)
print(
    f" P={P.mean().item():.6f} R={R.mean().item():.6f} F={F1.mean().item():.6f}"
)
