#!/usr/bin/env python3
"""Write revised role-aware annotations for the filtered candidate sample100."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_SAMPLE = "outputs/cross_domain_eval_selection/author_count3_jsd_l0_ge0p03186_highconf2/sample100_audit.jsonl"
DEFAULT_OUT = "outputs/cross_domain_eval_selection/author_count3_jsd_l0_ge0p03186_highconf2/sample100_role_internal_annotations.tsv"
DEFAULT_SUMMARY = "outputs/cross_domain_eval_selection/author_count3_jsd_l0_ge0p03186_highconf2/sample100_role_internal_annotations.summary.tsv"


ROLE_CLEAR = {
    "2858257636": "Edge-computing control in dense V2X combines mobile edge systems, wireless interference, latency, and energy optimization.",
    "2898980485": "Hybrid transportation safety task combines traffic mobility, public safety, situational awareness, and networking systems.",
    "2782254815": "Cortical microcircuit learning combines neural-network learning mechanisms with neuroscience/synaptic plasticity.",
    "2588149001": "Outlier-analysis visualization combines anomaly/statistical learning with visual analytics and human-judgment workflow.",
    "2963808864": "Stock prediction task combines finance/volatility, social-media emotion mining, causality, and predictive modeling.",
    "2727098759": "IoT energy challenge combines embedded software, energy-aware systems, and tooling/analysis concerns.",
    "2887425801": "Next-generation processors paper combines neural/graph algorithms with neuromorphic and quantum processor expertise.",
    "2949167152": "Linked-open-data distinctions combine ontology/semantic-web engineering with NLP and foundational ontology evidence.",
    "2788044590": "Large-class student support combines CS education, pedagogy, intervention design, and classroom evidence.",
    "2794386110": "Unpaired image captioning needs both visual representation and NLP/language-pivot expertise.",
    "2894475059": "Spiking neural network with visible-light communications combines neural models and optical/wireless communication systems.",
    "2786714961": "Hospital analytics task combines healthcare/patient satisfaction with information-systems analytics.",
    "2792896596": "Smart-home elderly QoL combines smart systems, technology acceptance, social/behavioral models, and ICT context.",
    "2895993965": "Dance-posture recognition combines domain-specific movement/dance posture, image matching, and evolutionary optimization.",
    "2623712993": "RFID activity recognition combines ubiquitous sensing, device-free signal processing, and ML/data mining.",
    "2786399470": "Sentiment-controlled image captioning combines affect/sentiment modeling with image captioning/vision.",
    "2776175080": "Electricity-market scheduling combines power systems/smart grid, game theory, and distributed optimization/control.",
    "2893667970": "Music salience/reduction combines music theory with computational modeling and MIR expertise.",
    "2796966199": "EEG/MEG plus multimodal MRI combines neuroimaging, signal modeling, and neuroscience expertise.",
    "2807008354": "Web cryptomining analysis combines web security, monetization/profitability, and user-experience cost measurement.",
    "2517282811": "Business logic in collaborative networks combines business rules/workflows, semantic web, and swarm/stigmergy coordination.",
    "2890476315": "Casual creator study combines game design, HCI, and computational creativity expertise.",
    "2789783607": "Edge-bundling parameter search combines visualization/geovisualization with evolutionary optimization.",
    "2740766427": "Influenza-like illness estimation combines web search/IR with public-health surveillance signal selection.",
    "2901025079": "TTS representation mixing combines speech synthesis/processing with neural representation learning.",
    "2365029783": "Encrypted metering-data query combines privacy/security, cloud/big-data storage, and indexing/search.",
    "2952303469": "Edge-caching content market combines wireless/content-service systems with game-theoretic market modeling.",
    "2889439452": "BPMN stochastic analysis combines business-process modeling with rewriting logic/formal methods.",
    "2891066506": "Blind-user navigation combines accessibility/HCI, smartphone guidance, and vision/navigation expertise.",
    "2883327901": "Multimodal anomaly detection combines statistical change detection, sensors/communications, and pattern recognition.",
    "2345200429": "HVAC energy management combines building/HVAC energy systems with optimization/control.",
    "2894040201": "Underwater magnetic-field mapping combines marine sensing, magnetometer calibration, and robotic mapping.",
    "2949269310": "Vehicle classification combines FMCW radar/remote sensing signals with convolutional-network classification.",
    "2897902636": "Flexible space robot manipulators combine spacecraft/robot control with nonparametric identification/ML.",
    "2793148261": "Medical dataset classification combines health data, evolutionary optimization, and parallel deep learning.",
    "2963614783": "Textual grounding links language words to image concepts, requiring vision-language and object-detection expertise.",
}


ROLE_INTERNAL = {
    "2789509531": "Within CV, the task separates active-contour segmentation, kernel descriptors, and optimization/learning expertise.",
    "2883104196": "Within data/semantic web, the task separates RDF/SPARQL querying, metadata stratification, and formal rewriting.",
    "2807921578": "Within IoT/security, the task separates WSN/IoT networking from cryptographic access-control protocol design.",
    "2604599596": "Within database/KR theory, the task separates ontology-mediated queries, existential rules, and decidability/containment.",
    "2522440390": "Within networked control, the task separates system observability/controllability, automata data-loss models, and control theory.",
    "2897390578": "Within requirements engineering, the task separates RE methods, smart-city/rural contexts, and societal/application framing.",
    "2890683554": "Within robotics, the task separates grasp-quality metrics, robot manipulation, and ML success prediction.",
    "2796056127": "Within audio ML, the task separates source separation, robust statistics, and k-NN/hubness analysis.",
    "2546190447": "Within vision, the task separates cross-modal scene data, CNN representation learning, and transfer across modalities.",
    "2909565146": "Within NLP/semantic systems, the task separates controlled natural language, semantic analysis, and knowledge-base creation.",
    "2884164506": "Within graphics/systems, the task separates rendering, cloud delivery, latency, and bandwidth constraints.",
    "2766476550": "Within edge/cloud systems, the task separates monitoring, requirements analysis, QoS/QoE, and workflow/cloud expertise.",
    "2801497650": "Within SAR imaging, the task separates radar imaging, sparse reconstruction, and computational signal processing.",
    "2793512125": "Within networked multi-agent modeling, the task separates opinion dynamics, Markov models, and control/game perspectives.",
    "2800124572": "Within event recognition, the task separates CV feature models, ensemble optimization, and application/video recognition.",
    "2964315715": "Within ML, the task separates autoregressive density modeling, anomaly detection, and optimization/statistical modeling.",
    "2812009592": "Within neural-network optimization, the task separates feature replay, convergence, and model training.",
    "2788725334": "Within data mining, the task separates heterogeneous networks/meta-paths, metric learning, and author identification.",
    "2763881998": "Within speech security, the task separates spoofing, speech features, and classification/evaluation.",
    "2898541470": "Within CV/ML, the task separates point-set registration, manifold regularization, and robust transformation learning.",
    "2770462575": "Within medical image interaction, the task separates segmentation, latency management, and visualization/CV.",
    "2963945023": "Within video restoration, the task separates deblurring, optical flow/image formation, and self-supervised learning.",
    "2765320984": "Within software/service processes, the task separates ticket workflow, process mining, service management, and SE.",
    "2621274099": "Within quantum information, the task separates entanglement distillation, capacity, and information-theoretic bounds.",
    "2895589658": "Within video person re-ID, the task separates unsupervised representation learning, manifold/embedding, and visual tracking.",
    "2798513548": "Within formal methods, the task separates runtime enforcement, process logic, and theoretical correctness.",
    "2617623622": "Within operations research, the task separates ship-routing domain constraints, speed optimization, and column generation.",
    "2885712064": "Within smart-home security, the task separates home automation, mutual authentication, and cryptographic protocol design.",
    "2738246058": "Within image compression, the task separates psychovisual thresholds, quantization/DCT, and bit-allocation strategy.",
    "2793314310": "Within cloud/network systems, the task separates service placement, QoS, and evolutionary optimization.",
    "2899699430": "Within network reliability, the task separates router-domain knowledge, anomaly detection, and feature/prediction modeling.",
    "2782305552": "Within distributed algorithms, the task separates graph coloring, dynamic networks, and formal proof/correctness.",
    "2807162623": "Within crypto/math, the task separates Lucas sequences, Gaussian integers, elliptic curves, and cryptanalysis.",
    "2962710293": "Within systems security, the task separates Linux kernel/OS memory safety and security/hardware roots.",
    "2901025079": "Within speech ML, the task separates TTS representation control, pronunciation/speech expertise, and neural encoders.",
    "2793651583": "Within embedded systems, the task separates microarchitecture, cache/compiler design, and WCET analysis.",
    "2776455340": "Within fuzzy/discrete math, the task separates fuzzy sets, graphs, and algebraic graph properties.",
    "424033818": "Within data warehousing/IS, the task separates source-data update propagation, stochastic methods, and business/data storage.",
    "2782994636": "Within cross-media retrieval, the task separates hashing, semantic consistency, and multiview media features.",
    "2317840569": "Within cloud architecture, the task separates brokerage, cloud services, architecture classification, and software systems.",
    "2900543929": "Within statistical shape modeling, the task separates active shape models, Bayesian inference, and MCMC fitting.",
    "2887459746": "Within cloud/agent systems, the task separates trust/reputation, cloud-of-things infrastructure, and group formation.",
    "2526794381": "Within anomaly detection, the task separates cluster structure, SVM/mixture modeling, and incremental detection.",
    "2963881501": "Within distributed statistical learning, the task separates social learning, hypothesis testing, and communication/network constraints.",
    "2600455691": "Within temporal database/KR, the task separates metric temporal logic, Datalog, ontology/data access, and query complexity.",
    "2897098626": "Within software development, the task separates user-interaction data, process objectives, and software engineering challenges.",
    "2884543123": "Within clustering/ML, the task separates ensemble clustering, pivot features, and algorithmic acceleration.",
    "2755572239": "Within hybrid systems, the task separates reachability, programming-language semantics, and CPS modeling.",
    "2592906030": "Within tracking/control, the task separates multi-sensor control, Bayes filters, and POMDP/optimization.",
    "2963229750": "Within mathematical ML, the task separates optimal transport, information geometry, and regularization.",
    "2893537147": "Within argumentation AI, the task separates logical argumentation graphs, workflow/generation, and benchmark construction.",
    "2585468005": "Within wireless propagation, the task separates UAV channel sounding, path loss, and air-to-ground channel modeling.",
    "2903147819": "Within control/optimization, the task separates Newton solvers, MPC, linear inequalities, and optimal control.",
    "2895102134": "Within ML privacy/security, the task separates deep generative models, membership attacks, and overfitting/inference.",
    "2897151058": "Within quantum crypto, the task separates quantum information, cryptographic primitives, and hardware assumptions.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-tsv", default=DEFAULT_OUT)
    parser.add_argument("--summary-tsv", default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for index, raw in enumerate(Path(args.sample_jsonl).open("r", encoding="utf-8"), 1):
        obj = json.loads(raw)
        paper_id = str(obj.get("paper_id") or obj.get("id"))
        if paper_id in ROLE_CLEAR:
            label = "role_based_clear"
            reason = ROLE_CLEAR[paper_id]
            usable = 1
        elif paper_id in ROLE_INTERNAL:
            label = "role_based_internal"
            reason = ROLE_INTERNAL[paper_id]
            usable = 1
        else:
            label = "weak_or_common_interest"
            reason = "Facets and author histories look too concentrated in one method/community, or role division is not clear enough from available evidence."
            usable = 0
        rows.append(
            {
                "sample_index": index,
                "paper_id": paper_id,
                "title": str(obj.get("title") or ""),
                "author_count": len(obj.get("authors") or []),
                "manual_label": label,
                "role_usable_for_stage1": usable,
                "reason": reason,
            }
        )

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["manual_label"] for row in rows)
    usable = sum(int(row["role_usable_for_stage1"]) for row in rows)
    with Path(args.summary_tsv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["metric", "value"])
        for key in ("role_based_clear", "role_based_internal", "weak_or_common_interest"):
            writer.writerow([key, counts[key]])
        writer.writerow(["role_usable_for_stage1", usable])
        writer.writerow(["sample_size", len(rows)])
    print(f"wrote={out_path}")
    print(f"wrote={args.summary_tsv}")
    print(dict(counts), f"usable={usable}")


if __name__ == "__main__":
    main()
