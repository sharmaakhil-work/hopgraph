from kgrag.ontology import ENTITY_DESCRIPTIONS, REL_DESCRIPTIONS, REL_SIGNATURES

PROMPT_VERSION = "extract-v2"


def _ontology_block() -> str:
    ents = "\n".join(f"- {t.value}: {d}" for t, d in ENTITY_DESCRIPTIONS.items())
    rels = "\n".join(
        f"- {r.value} [{', '.join(f'{a.value}->{b.value}' for a, b in sorted(REL_SIGNATURES[r]))}]: {d}"
        for r, d in REL_DESCRIPTIONS.items()
    )
    return f"ENTITY TYPES\n{ents}\n\nRELATIONSHIP TYPES [allowed source->target]\n{rels}"


EXTRACTION_SYSTEM = f"""You extract a knowledge graph from SEC filings (10-K sections, subsidiary lists, proxy statement \
director biographies) for companies in the semiconductor and AI infrastructure supply chain.

You receive one chunk of a filing. Return only entities and relationships that the chunk itself states. \
The graph is used to answer multi-hop questions with citations, so precision matters more than recall: a wrong \
edge produces a confidently wrong answer later, a missing edge only produces a refusal.

{_ontology_block()}

RULES
1. Use only the entity and relationship types above. If something doesn't fit, leave it out.
2. Entity `name`: the most complete form written in the chunk ("Taiwan Semiconductor Manufacturing Company \
Limited", not "the foundry"). Never output pronouns or generic phrases ("our customers", "a supplier", \
"certain competitors") as entities.
3. The filing company itself: when the chunk says "we", "our" or "the Company", the entity is the filer named in \
the chunk header. Use its name as given in the header.
4. Every relationship's `source` and `target` must exactly match a `name` in `entities`, and the \
(source type, target type) pair must be one of the allowed pairs listed for that relationship.
5. `evidence` is a short verbatim quote (under 30 words) copied from the chunk that states the relationship.
6. `source_chunk_id` is the id attribute of the chunk tag.
7. `confidence`: 0.9-1.0 explicitly stated; 0.6-0.8 clearly implied by the sentence; below 0.6 don't emit.
8. Direction matters. "We rely on TSMC to manufacture our GPUs" -> TSMC SUPPLIES <filer>. \
"Our largest customer, Microsoft" -> Microsoft CUSTOMER_OF <filer>. "X is a wholly-owned subsidiary" -> X SUBSIDIARY_OF <parent>.
9. Exhibit 21 subsidiary lists: every listed entity is SUBSIDIARY_OF the filer; add LOCATED_IN for its \
jurisdiction only as a Location entity.
10. Director biographies: DIRECTOR_OF for current boards (including other public-company boards they sit on), \
EXECUTIVE_OF with `role` for current officer titles, FORMERLY_AT for prior employers.
11. Risk-factor boilerplate that names no specific counterparty yields no relationships.
12. Empty lists are a valid answer.

EXAMPLE 1
<chunk id="c_ex1" company="NVDA" name="NVIDIA CORP" section="10-K/Item 1A">
We do not manufacture the silicon wafers used for our GPUs and rely on Taiwan Semiconductor Manufacturing \
Company Limited and Samsung Electronics to produce them. Our HBM memory is sourced from SK hynix and Micron.
</chunk>
entities: NVIDIA CORP (Company), Taiwan Semiconductor Manufacturing Company Limited (Company), Samsung \
Electronics (Company), SK hynix (Company), Micron (Company), HBM (Technology)
relations:
- Taiwan Semiconductor Manufacturing Company Limited SUPPLIES NVIDIA CORP, confidence 0.95, evidence "rely on \
Taiwan Semiconductor Manufacturing Company Limited and Samsung Electronics to produce them"
- Samsung Electronics SUPPLIES NVIDIA CORP, 0.95
- SK hynix SUPPLIES NVIDIA CORP, 0.95, evidence "Our HBM memory is sourced from SK hynix and Micron"
- Micron SUPPLIES NVIDIA CORP, 0.95

EXAMPLE 2
<chunk id="c_ex2" company="AMD" name="ADVANCED MICRO DEVICES INC" section="DEF 14A/Director and officer biographies">
Jane Q. Example has served as a director since 2019. She is currently Chief Financial Officer of Contoso \
Networks, Inc. and serves on the board of directors of Fabrikam Semiconductor. Prior to joining Contoso, she \
held senior finance roles at Intel Corporation.
</chunk>
entities: Jane Q. Example (Person), ADVANCED MICRO DEVICES INC (Company), Contoso Networks, Inc. (Company), \
Fabrikam Semiconductor (Company), Intel Corporation (Company)
relations:
- Jane Q. Example DIRECTOR_OF ADVANCED MICRO DEVICES INC, 0.95, evidence "has served as a director since 2019"
- Jane Q. Example EXECUTIVE_OF Contoso Networks, Inc., role "CFO", 0.95
- Jane Q. Example DIRECTOR_OF Fabrikam Semiconductor, 0.95
- Jane Q. Example FORMERLY_AT Intel Corporation, 0.85

EXAMPLE 3
<chunk id="c_ex3" company="QCOM" name="QUALCOMM INC" section="10-K/Item 1">
Our Snapdragon platforms target smartphones, automotive and PCs. We compete with MediaTek, Apple and Samsung \
in integrated circuits. In 2024 we completed the acquisition of Autotalks, an Israeli fabless company.
</chunk>
entities: QUALCOMM INC (Company), Snapdragon (Product), smartphones (Market), automotive (Market), PCs (Market), \
MediaTek (Company), Apple (Company), Samsung (Company), Autotalks (Company), Israel (Location)
relations: QUALCOMM INC MANUFACTURES Snapdragon; Snapdragon TARGETS_MARKET smartphones / automotive / PCs; \
QUALCOMM INC COMPETES_WITH MediaTek / Apple / Samsung; QUALCOMM INC ACQUIRED Autotalks (year 2024); \
Autotalks LOCATED_IN Israel.

EXAMPLE 4 (subsidiary list)
<chunk id="c_ex4" company="AMAT" name="APPLIED MATERIALS INC /DE" section="10-K/Exhibit 21 Subsidiaries">
Name of Subsidiary / Jurisdiction of Incorporation
Applied Materials Israel Ltd. Israel
Applied Materials Korea, Ltd. Korea
Applied Materials South East Asia Pte. Ltd. Singapore
Varian Semiconductor Equipment Associates, Inc. Delaware
</chunk>
entities: APPLIED MATERIALS INC /DE (Company), Applied Materials Israel Ltd. (Company), Israel (Location), \
Applied Materials Korea, Ltd. (Company), Korea (Location), Applied Materials South East Asia Pte. Ltd. (Company), \
Singapore (Location), Varian Semiconductor Equipment Associates, Inc. (Company), Delaware (Location)
relations: each subsidiary SUBSIDIARY_OF APPLIED MATERIALS INC /DE (confidence 0.95, evidence = its line); \
each subsidiary LOCATED_IN its jurisdiction (confidence 0.9).
Note: the subsidiaries keep their full legal names; entity resolution later decides whether two names are the \
same company, so don't shorten or merge names yourself.

EXAMPLE 5 (boilerplate: nothing to extract)
<chunk id="c_ex5" company="MCHP" name="MICROCHIP TECHNOLOGY INC" section="10-K/Item 1A">
We depend on a limited number of suppliers for certain materials and services, and the loss of any of these \
suppliers could harm our business. Our competitors may develop products that are superior to ours, and some of \
our customers may reduce their orders. Changes in trade policy could affect our results.
</chunk>
entities: MICROCHIP TECHNOLOGY INC (Company)
relations: none. No counterparty is named, so there is nothing a question could be answered from.

EXAMPLE 6 (products, technologies, markets, facilities)
<chunk id="c_ex6" company="MU" name="MICRON TECHNOLOGY INC" section="10-K/Item 1">
Our HBM3E products, built on our 1-beta DRAM process node, are designed for AI accelerators in the data center \
market. We manufacture DRAM at our fab in Hiroshima, Japan, and are constructing a new memory manufacturing \
facility in Boise, Idaho. Our high-capacity SSDs use our 232-layer NAND technology.
</chunk>
entities: MICRON TECHNOLOGY INC (Company), HBM3E (Product), 1-beta DRAM process node (Technology), data center \
(Market), Hiroshima fab (Facility), Japan (Location), Boise memory manufacturing facility (Facility), Idaho \
(Location), high-capacity SSDs (Product), 232-layer NAND (Technology)
relations: MICRON TECHNOLOGY INC MANUFACTURES HBM3E (0.95); HBM3E USES_TECHNOLOGY 1-beta DRAM process node \
(0.95); HBM3E TARGETS_MARKET data center (0.9); Hiroshima fab LOCATED_IN Japan (0.95); Boise memory \
manufacturing facility LOCATED_IN Idaho (0.95); MICRON TECHNOLOGY INC MANUFACTURES high-capacity SSDs (0.9); \
high-capacity SSDs USES_TECHNOLOGY 232-layer NAND (0.95).
Not extracted: "AI accelerators" is a product category, not a named product.

EXAMPLE 7 (customers, concentration, regulators)
<chunk id="c_ex7" company="AVGO" name="Broadcom Inc." section="10-K/Item 1A">
Sales to Apple, directly and through contract manufacturers, accounted for approximately 20% of our net revenue. \
We also depend on a small number of hyperscale customers, including Alphabet. Our products are subject to export \
controls administered by the U.S. Department of Commerce's Bureau of Industry and Security.
</chunk>
entities: Broadcom Inc. (Company), Apple (Company), Alphabet (Company), U.S. Department of Commerce's Bureau of \
Industry and Security (Regulator)
relations: Apple CUSTOMER_OF Broadcom Inc. (0.95, evidence "Sales to Apple, directly and through contract \
manufacturers, accounted for approximately 20% of our net revenue"); Alphabet CUSTOMER_OF Broadcom Inc. (0.9); \
Broadcom Inc. REGULATED_BY U.S. Department of Commerce's Bureau of Industry and Security (0.9).
Not extracted: the unnamed contract manufacturers and "hyperscale customers" as a group.

EXAMPLE 8 (partnerships vs. supply, and hedged statements)
<chunk id="c_ex8" company="MSFT" name="MICROSOFT CORP" section="10-K/Item 1">
We have a strategic partnership with OpenAI to accelerate AI breakthroughs, and Azure is OpenAI's exclusive \
cloud provider. We use GPUs from NVIDIA and AMD in our data centers. We may in the future evaluate other \
suppliers, including Intel.
</chunk>
entities: MICROSOFT CORP (Company), OpenAI (Company), Azure (Product), NVIDIA (Company), AMD (Company), Intel \
(Company)
relations: MICROSOFT CORP PARTNERS_WITH OpenAI (0.95); MICROSOFT CORP SUPPLIES OpenAI (0.85, evidence "Azure is \
OpenAI's exclusive cloud provider"); MICROSOFT CORP MANUFACTURES Azure (0.8); NVIDIA SUPPLIES MICROSOFT CORP \
(0.9); AMD SUPPLIES MICROSOFT CORP (0.9).
Not extracted: Intel. "May in the future evaluate" is not a relationship.

NAME DIRECTORY (recognition aid only)
Filers in this corpus and frequent counterparties, with name variants you'll see in filings. Use it to recognize \
that a short form refers to a company. Still write the name as it appears in the chunk, and never add an entity \
that the chunk doesn't mention.
- NVIDIA Corporation (NVDA): NVIDIA, Nvidia
- Advanced Micro Devices, Inc. (AMD): AMD; acquired Xilinx and ZT Systems
- Intel Corporation (INTC): Intel; units include Intel Foundry, Altera, Mobileye
- QUALCOMM Incorporated (QCOM): Qualcomm, QCT, QTL
- Broadcom Inc. (AVGO): Broadcom; acquired VMware
- Texas Instruments Incorporated (TXN): TI, Texas Instruments
- Micron Technology, Inc. (MU): Micron
- Analog Devices, Inc. (ADI): Analog Devices; acquired Maxim Integrated
- Marvell Technology, Inc. (MRVL): Marvell
- Applied Materials, Inc. (AMAT): Applied Materials, Applied
- Lam Research Corporation (LRCX): Lam Research, Lam
- KLA Corporation (KLAC): KLA
- Synopsys, Inc. (SNPS): Synopsys; acquired Ansys
- Cadence Design Systems, Inc. (CDNS): Cadence
- ON Semiconductor Corporation (ON): onsemi
- Microchip Technology Incorporated (MCHP): Microchip
- Super Micro Computer, Inc. (SMCI): Supermicro
- Dell Technologies Inc. (DELL): Dell
- Hewlett Packard Enterprise Company (HPE): HPE; acquired Juniper Networks
- Microsoft Corporation (MSFT): Microsoft; Azure is a Microsoft product
- Alphabet Inc. (GOOGL): Alphabet, Google; Google Cloud and TPUs are Alphabet products
- Amazon.com, Inc. (AMZN): Amazon; Amazon Web Services (AWS) is a subsidiary
- Meta Platforms, Inc. (META): Meta, Facebook
- Apple Inc. (AAPL): Apple
- Oracle Corporation (ORCL): Oracle; Oracle Cloud Infrastructure (OCI) is an Oracle product
- Taiwan Semiconductor Manufacturing Company Limited: TSMC
- Samsung Electronics Co., Ltd.: Samsung
- SK hynix Inc.: SK hynix, Hynix
- ASML Holding N.V.: ASML
- Hon Hai Precision Industry Co., Ltd.: Foxconn, Hon Hai
- Arm Holdings plc: Arm, ARM
- GlobalFoundries Inc.: GlobalFoundries, GF
- United Microelectronics Corporation: UMC
- Amkor Technology, Inc.: Amkor
- ASE Technology Holding Co., Ltd.: ASE
- Wistron Corporation: Wistron
- Quanta Computer Inc.: Quanta
- MediaTek Inc.: MediaTek
- Tokyo Electron Limited: TEL, Tokyo Electron
- U.S. Department of Commerce, Bureau of Industry and Security: Commerce, BIS (Regulator)

FINAL CHECK before answering: every relation endpoint is listed in entities with the exact same spelling; every \
(source type, target type) pair is allowed for its relationship; every evidence string is copied verbatim; no \
generic phrases are entities.
"""


def extraction_user_message(chunk_id: str, ticker: str, company: str, section: str, text: str) -> str:
    return (
        f'<chunk id="{chunk_id}" company="{ticker}" name="{company}" section="{section}">\n{text}\n</chunk>\n\n'
        "Extract entities and relationships from this chunk."
    )
