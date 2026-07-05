#!/usr/bin/env python3
"""Write manual cross-domain collaboration annotations for smoke_200."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SAMPLE = "outputs/stage1_pilot_samples/smoke_200.jsonl"
DEFAULT_OUT = "outputs/cross_domain_eval_selection_smoke_200/smoke_200_manual_cross_domain_annotations.tsv"


YES_REASONS = {
    "2888060193": "Mobility services combine service reliability, business/payment processes, automation, and system-of-systems coordination.",
    "2802682160": "Clinical body-weight/health monitoring task using RGB-D sensing and patient-state estimation.",
    "2587693758": "Driver drowsiness estimation combines EEG/BCI signal evidence with active-learning regression.",
    "2571862639": "Building loads, energy storage, smart grid, and demand response indicate power systems plus building/end-user energy management.",
    "2801292404": "Virtual prototyping of heterogeneous cyber-physical systems combines modeling, software engineering, and system simulation.",
    "2890377281": "Microelectronic chip packaging study couples transient heat flow, electric currents, numerical analysis, and microelectronics.",
    "2903145669": "Information-systems theory paper explicitly combines causality, ontology/epistemology, and IS theorizing.",
    "2821617508": "Radar-cellular coexistence combines radar sensing, cellular communications, interference, and statistical testing.",
    "2797670873": "Medical ultrasound segmentation combines clinical joint anatomy with image segmentation and clustering.",
    "2810333617": "Language service infrastructure combines web services, multilingual language resources, usability, and maintenance engineering.",
    "2610406656": "Heteroscedastic regression/active learning is paired with humanoid affordance and robot-learning expertise.",
    "2622234068": "Railway rolling-stock application is paired with assignment, integer programming, greedy, and local-search optimization.",
    "2783848321": "Software product-line configuration combines recommender systems, ERP/context modeling, mass customization, and software engineering.",
    "2893300104": "Robot/visual-sensor fusion task combines computer vision, GPS, probability matching, and inference.",
    "2806190243": "Cross-lingual mathematical terminology extraction combines NLP, semantic search, clustering, and mathematical text.",
    "2795676171": "Human listening-rate study combines accessibility/screen readers, speech synthesis, crowdsourcing, and HCI.",
    "2781735277": "FPGA acceleration of deep learning combines RTL/compiler/hardware acceleration with neural-network workloads.",
    "2802196416": "Mobile-edge named-data-networking security combines edge/network systems, authentication, cryptography, and scalability.",
    "2894229479": "Cross-company value modeling combines conceptual/value-network modeling, enterprise planning, and empirical study design.",
    "2811319304": "Fog capacity provisioning combines network architecture, QoS/cloudlets, queueing, and optimization.",
    "2790874855": "Industrial application-layer firewall combines cybersecurity, industrial control systems, Modbus, and performance testbeds.",
    "2792778837": "Fisheries marketing cooperative combines game theory/economics with common-pool resource and commercialization context.",
    "2803729172": "Stochastic image and shape matching combines stochastic differential/statistical inference with image/shape analysis.",
    "2963358464": "Traffic forecasting combines transportation/traffic-flow modeling with graph recurrent deep learning.",
    "2801562232": "Financial disclosure reaction combines finance/market labels with topic modeling and text analytics.",
    "2790404719": "Building energy management combines MPC/regression trees, demand response, cyber-physical systems, and energy management.",
    "2810666255": "Molecular communication is framed as a biological system, combining communication theory and biology.",
    "2794891311": "Aerial inspection combines visual-inertial robotics, vision, trajectory, and inspection application needs.",
    "2919409811": "Distributed programming over LoRaWAN for CPS combines wireless/IoT networking, software architecture, and intelligent transportation.",
    "2887633198": "Business-value modeling comparison combines IS/business value, empirical experiments, software, and usability.",
    "2795686945": "Vehicular check-in combines intelligent transportation/traffic-flow labels with wireless networking and simulation.",
    "2888049133": "Green IS adoption study combines information systems, sustainability/green computing, finance, and marketing context.",
    "2897494692": "Asset allocation via sentiment combines finance/portfolio theory, market sentiment, NLP sentiment analysis, and ML.",
    "2734648576": "Power-grid monitoring combines electric power systems, phasor/frequency estimation, Fourier/harmonic analysis, and statistics.",
    "2884003968": "Fairness definition combines differential privacy/fairness with intersectionality and social categories.",
    "2890811531": "Envy-free classification combines fair division/social-choice style constraints with classifier design.",
    "2784015601": "Interactive image segmentation is explicitly framed with feedback-control perspective and image/medical segmentation.",
    "2902386461": "Medical-record diagnosis support combines healthcare, NLP/RDF, data integration, and decision support.",
    "2792586211": "Multilateral teleoperation survey combines robotics, control, network topology, and teleoperation task analysis.",
    "2901284906": "SAR image segmentation combines remote sensing imagery, Bayesian/HMM fusion, discriminative models, and segmentation.",
    "2897540526": "IoT speed-up paper combines lightweight encryption/security with low-latency IoT communications.",
    "2864245682": "Industrial wireless sensor/actuator testbed combines industrial systems, wireless networking, synchronization, latency, and testbed work.",
    "2911943722": "FlexRay security exploits combine automotive in-vehicle protocols, physical/protocol layers, and cybersecurity.",
    "2465806284": "Smart-grid load balancing combines demand response/load management with optimization and exact load-balancing methods.",
    "2795345736": "Industrial business-process simulation combines business process management, metamodeling, software engineering, and information science.",
}


BORDERLINE_REASONS = {
    "2897410780": "Adaptive learning, temporal data, Twitter, and statistical testing are multi-facet, but mostly one ML evaluation setting.",
    "2783983795": "NLP-style ambiguity detection is applied to software variation points; plausible cross-area but evidence is limited.",
    "2807869908": "Mobile energy savings combines supervisory control and mobile systems, but remains close to systems/control.",
    "2932849458": "DNN compression, energy constraints, and bilinear regression are distinct facets but mostly within efficient deep learning.",
    "2767662337": "Conformance analysis links data and process perspectives, but the topic remains close to process mining.",
    "2765345348": "Semantic task-failure prevention in robotics combines ontology and robot planning, but evidence is sparse.",
    "2887051655": "Multi-agent middleware covers software architecture and building/cognitive labels, but still a software-architecture paper.",
    "2790808553": "Generation control, distributed generation, and economic dispatch are split across authors but within power-system control.",
    "2883900035": "NMT with SMT word knowledge is multi-method NLP rather than clearly cross-domain.",
    "2963881246": "Two-unicast/network coding is technical but mostly one information-theory/network-coding area.",
    "2963472766": "Full-duplex communications and self-interference optimization are internal wireless-communications facets.",
    "2914343252": "Higher-order network visualization/analysis is multi-facet but mostly network-analysis work.",
    "2963937708": "Colonel Blotto combines game theory and resource allocation, but it is mainly mathematical economics/game theory.",
    "2888928288": "Ethereum exploit analysis combines blockchain and security, but still one security/program-analysis niche.",
    "2895157071": "Formal-methods labels are distributed, but the paper remains within logic/model checking/theoretical CS.",
    "2761857866": "Cognitive heterogeneous networks include several channel/network facets but stay inside wireless communications.",
    "2914291999": "Robust self-oscillation uses control and nonlinear analysis but remains control theory.",
    "2614851539": "V2I beam alignment has vehicular/wireless positioning facets, but most labels remain wireless communications.",
    "2799041781": "Cache-aided multicast involves caching and coded multicast, but mainly network information theory.",
    "2789103234": "Millimeter-wave coordination combines optimization/topology/interference but stays in wireless networking.",
    "2891155820": "Source localization combines audio/noise/features/covariance, but mostly signal processing.",
    "2964318145": "Distributed constrained optimization shows author dispersion, but the content remains optimization/consensus.",
    "2892143093": "HEVC tiling plus processor assignment mixes video encoding and scheduling, but still an implementation optimization paper.",
    "2547467263": "IEEE 802.11 heterogeneous throughput optimization is mostly networking/control.",
    "2800298750": "Arduino code generation from modeling language has software-plus-embedded flavor, but evidence is thin.",
    "2963698335": "LiFi throughput maximization has network/optical-wireless facets but remains communications.",
    "2963483933": "Finite sets/cardinality constraints in SMT are all formal-methods/theory facets.",
    "2765183019": "Distributed linked-data archives combine web, preservation, and querying, but stay within data/web infrastructure.",
    "2964104099": "Exploratory-testing automation combines software testing and model reconstruction, but remains software engineering.",
    "2962985009": "Dynamic fault trees with GSPNs combines reliability and Petri nets, but remains formal dependability modeling.",
    "2888559858": "Vehicular-network connectivity/service discovery is networking-internal.",
    "2782367542": "Inverse source problem combines PDE/frequency/attenuation mathematics, but remains applied math.",
    "2508101221": "MapReduce acceleration with SSDs combines storage and distributed systems, but remains computer systems.",
    "2804694832": "Matrix completion with neural networks combines matrix methods and NN optimization, but remains ML/numerical methods.",
    "2963590174": "Inverse problems with deep/compressed sensing tradeoffs are multi-method but still inverse-problem methodology.",
    "2807545501": "Context-aware crowdsourcing/zoning services have social/spatiotemporal facets, but evidence is moderate.",
    "2792027525": "Android malware explanation combines security and interpretability/ML but remains malware analysis.",
    "2795409017": "Filtering on Lie groups combines control/filtering and geometry, but the scope is narrow.",
    "2404048309": "Semantic web ontology construction has ontology/search/semantic labels but remains knowledge engineering.",
    "2796431263": "Graph signal processing spans graph, signal, image, and sensor applications, but much is within one emerging area.",
    "2766402906": "Decision-tree optimization combines ML and multi-objective optimization, but it is mostly a method paper.",
    "2889227180": "Privacy-protected databases combine inference, encryption, and database privacy, but evidence is moderate.",
    "2963912371": "Sparse Kerdock neighbor discovery combines wireless and coding/matrix methods, but likely one communications/coding niche.",
    "2788152782": "Rank aggregation with manifold learning mixes ranking and nonlinear dimensionality reduction, but mostly ML.",
    "2891171329": "RAN resource management with learning is multi-facet but remains wireless networking.",
    "2787381105": "Privacy leakage in multi-agent planning combines privacy and planning, but evidence is not strong enough for yes.",
    "2810073886": "Multi-agent path finding with destructible obstacles is multi-agent robotics/planning but not clearly cross-domain.",
    "2575486826": "Non-Lipschitz nonlinear programming is pure optimization/math despite multiple mathematical labels.",
    "2899176198": "Parametricity and universal type with crypto terms is theory/programming-languages adjacent, not clearly cross-domain.",
    "2794717842": "Lock-free indexing in non-volatile memory combines data structures and memory systems but remains systems.",
    "2778462592": "Ultra-dense networks with spatial spectrum reuse is wireless-networking internal.",
    "2789825276": "Lattice sieving with cryptographic labels is mostly computational lattice/crypto theory.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-tsv", default=DEFAULT_OUT)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, obj in enumerate(iter_jsonl(Path(args.sample_jsonl)), 1):
        paper_id = str(obj.get("id"))
        if paper_id in YES_REASONS:
            label = "yes"
            use_for_eval = 1
            reason = YES_REASONS[paper_id]
        elif paper_id in BORDERLINE_REASONS:
            label = "borderline"
            use_for_eval = 0
            reason = BORDERLINE_REASONS[paper_id]
        else:
            label = "no"
            use_for_eval = 0
            reason = "Single-domain, same-method family, too generic, or insufficient evidence of distinct cross-domain author roles from title/FoS."
        rows.append(
            {
                "sample_index": index,
                "paper_id": paper_id,
                "title": str(obj.get("title") or ""),
                "author_count": len(obj.get("authors") or []),
                "manual_label": label,
                "use_for_cross_domain_eval": use_for_eval,
                "reason": reason,
            }
        )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["manual_label"] for row in rows)
    print(f"out={out_path}")
    print(f"rows={len(rows)} yes={counts['yes']} borderline={counts['borderline']} no={counts['no']}")


if __name__ == "__main__":
    from collections import Counter

    main()
