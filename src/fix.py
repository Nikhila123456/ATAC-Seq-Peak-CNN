"""Run this from inside the atac_cnn folder: python fix.py"""
lines = open("evaluate.py").readlines()

for i, line in enumerate(lines):
    if ".cpu().numpy()" in line and "sigmoid" in line:
        lines[i] = line.replace(".cpu().numpy()", ".detach().cpu().numpy()")
    if "labels.numpy()" in line:
        lines[i] = line.replace("labels.numpy()", "labels.detach().cpu().numpy()")

open("src/evaluate.py", "w").writelines(lines)
print("Fixed. Run: python train_model.py")
