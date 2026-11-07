PAPER = "arXiv:2404.00498v2, Table 6"
PAPER_RUNS_PER_CELL = 400
PAPER_CI95_HALFWIDTH = 0.014

TABLE6 = {
    (10, False): {"none": 92.3053, "random": 93.0988, "alternating": 93.2798},
    (20, False): {"none": 92.8166, "random": 93.7446, "alternating": 93.8652},
    (40, False): {"none": 93.0143, "random": 94.0133, "alternating": 94.0729},
    (80, False): {"none": 93.0612, "random": 94.1169, "alternating": 94.1628},
    (10, True): {"none": 93.4071, "random": 93.9488, "alternating": 94.0186},
    (20, True): {"none": 93.8528, "random": 94.5565, "alternating": 94.6530},
    (40, True): {"none": 94.0381, "random": 94.7803, "alternating": 94.8203},
    (80, True): {"none": 94.0638, "random": 94.8506, "alternating": 94.8676},
}


def paper_value(epochs, tta, flip):
    row = TABLE6.get((int(epochs), bool(tta))) if float(epochs).is_integer() else None
    return None if row is None else row.get(str(flip))
