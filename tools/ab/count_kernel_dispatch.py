import csv, sys, collections
path = sys.argv[1]
names = collections.Counter()
with open(path, newline="") as f:
    rd = csv.DictReader(f)
    col = next((c for c in rd.fieldnames if c and "Kernel_Name" in c), None)
    for row in rd:
        names[row[col]] += 1
watch = ["quantize_images_kernel", "quantize_q8_0_kernel", "quantize_q8_K_kernel",
         "bf16_gemv_pair_warp_kernel", "bf16_gemv_warp_kernel", "pair_kernel", "swilu_quantize"]
print("file:", path, "| total dispatches:", sum(names.values()), "| distinct kernels:", len(names))
for w in watch:
    hits = {k: v for k, v in names.items() if w in k}
    print(f"  {w:30s} -> {sum(hits.values()):7d}", ("  " + "; ".join(f"{k[:60]}={v}" for k, v in hits.items())[:200]) if hits else "")
