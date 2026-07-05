#!/usr/bin/env python3
"""Write manual role-based-vs-common-interest annotations for a whole-validation sample."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_SAMPLE = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/full_fos_sample100_audit.jsonl"
DEFAULT_OUT = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/sample100_role_based_collaboration_annotations.tsv"
DEFAULT_SUMMARY = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/sample100_role_based_collaboration_annotations.summary.tsv"


ROLE_BASED_REASONS = {
    "2787507116": "VR headset realism needs display/rendering/perception expertise plus mobile/wireless systems constraints; author histories separate VR/mobile and wireless sensing/networking.",
    "2884064289": "Railway interlocking verification pairs railway-signalling domain knowledge with formal verification/model-checking expertise.",
    "2765951814": "Needle insertion paper combines medical brachytherapy/soft-tissue procedure knowledge with robotics, haptics, and control expertise.",
    "2804229147": "Federated heterogeneous IoT analytics combines IoT/platform integration, semantic interoperability, visual/data analytics, and web-accessibility/HCI histories.",
    "2752788051": "Technology intervention for everyday cognitive failure combines HCI/cognitive-failure framing with mobile/ubiquitous sensing and systems histories.",
    "2914648327": "OWL RL reasoner for Gene Ontology activity models combines semantic-web/reasoning work with biomedical ontology and bioinformatics expertise.",
    "2790894931": "Critical-infrastructure IoT resilience combines embedded sensing, signal/sensor systems, situational awareness, and healthcare/critical-infrastructure contexts.",
    "2783842267": "Gene-expression biclustering combines AI/evolutionary computation, algorithmic complexity, and bioinformatics/genetics expertise.",
    "2777149666": "Crisis informatics design-fiction work combines social media/HCI, disaster/open-data practice, crowdsourcing, and sociotechnical crisis-response expertise.",
    "2609835250": "Body-sensor and multilead biopotential bus design combines wearable/embedded electronics, sensor-network communication, and biomedical measurement constraints.",
    "2809305540": "Morphed-face fraud detection combines biometrics/face analysis, image forensics/watermarking, and border-document security context.",
}


BORDERLINE_REASONS = {
    "2768611354": "Information-security behavior is linked to cultural/psychological theory, but both authors' histories are close to information security and IS behavior.",
    "2887438492": "Mathematics-learning support has education and technology facets, but author histories do not cleanly separate into task-specific roles.",
    "2895736088": "Ontology evolution involves ontology engineering and knowledge representation, but the paper appears to stay within one knowledge-management/KR area.",
    "2610106470": "Networked predictive control mixes stochastic control with erasure-channel communication, but all histories are close to control/optimization/networked control.",
    "2794752269": "RPKI deployment has networking, security, and scalability facets, but the paper lacks enough content to confirm distinct author roles.",
    "2892418988": "Mobile-device data analysis suggests security/mobile/ML forensics, but the available evidence is too broad and author roles are not clear.",
    "2962989424": "Reactive synthesis with randomness touches formal methods, probabilistic behavior, and fuzz testing, but remains mostly theoretical CS/formal synthesis.",
    "2884432403": "Coupled neuron dynamics has biological-neuron and nonlinear-dynamics facets, but evidence for separate biology versus modeling roles is weak.",
    "2803928381": "AlphaZero in continuous action spaces draws on RL, tree search, and control/robotics examples, but it reads mostly as an AI-method paper.",
    "2181889564": "Spherical wavelets connect mathematical signal processing and cosmology-style applications, but the contribution is primarily mathematical.",
    "2807988936": "Ischaemia-potential sensitivity analysis is biomedical modeling, but both authors' histories are mainly numerical/mathematical modeling.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-tsv", default=DEFAULT_OUT)
    parser.add_argument("--summary-tsv", default=DEFAULT_SUMMARY)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def default_common_reason(obj: dict) -> str:
    if len(obj.get("authors") or []) <= 1:
        return "Single-author paper, so it cannot be evidence of team role division."
    return "Paper topic and author histories are concentrated in one technical community; collaboration looks closer to shared research interest than separable roles."


def main() -> None:
    args = parse_args()
    rows = []
    for index, obj in enumerate(iter_jsonl(Path(args.sample_jsonl)), 1):
        paper_id = str(obj.get("paper_id") or obj.get("id"))
        if paper_id in ROLE_BASED_REASONS:
            label = "role_based"
            role_based = 1
            reason = ROLE_BASED_REASONS[paper_id]
        elif paper_id in BORDERLINE_REASONS:
            label = "borderline"
            role_based = 0
            reason = BORDERLINE_REASONS[paper_id]
        else:
            label = "common_interest"
            role_based = 0
            reason = default_common_reason(obj)
        rows.append(
            {
                "sample_index": index,
                "paper_id": paper_id,
                "title": str(obj.get("title") or ""),
                "author_count": len(obj.get("authors") or []),
                "direct_l2_count": obj.get("direct_l2_count", ""),
                "manual_label": label,
                "is_role_based_collaboration": role_based,
                "reason": reason,
            }
        )

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["manual_label"] for row in rows)
    author_counts = Counter()
    for row in rows:
        if int(row["author_count"]) <= 1:
            author_counts["single_author"] += 1
        else:
            author_counts["multi_author"] += 1

    summary_rows = [
        {"metric": "sample_size", "value": len(rows)},
        {"metric": "role_based", "value": counts["role_based"]},
        {"metric": "borderline", "value": counts["borderline"]},
        {"metric": "common_interest", "value": counts["common_interest"]},
        {"metric": "single_author", "value": author_counts["single_author"]},
        {"metric": "multi_author", "value": author_counts["multi_author"]},
    ]
    summary_path = Path(args.summary_tsv)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"out={out_path}")
    print(f"summary={summary_path}")
    print(" ".join(f"{key}={counts[key]}" for key in ("role_based", "borderline", "common_interest")))
    print(f"single_author={author_counts['single_author']} multi_author={author_counts['multi_author']}")


if __name__ == "__main__":
    main()
