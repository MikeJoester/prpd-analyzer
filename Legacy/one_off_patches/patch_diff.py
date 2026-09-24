import re

with open("patchcore-inspection/src/patchcore/sampler.py", "r") as f:
    content = f.read()

new_func = """    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        a_times_a = (matrix_a ** 2).sum(dim=1, keepdim=True)
        b_times_b = (matrix_b ** 2).sum(dim=1).unsqueeze(0)
        a_times_b = matrix_a.mm(matrix_b.T)
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()"""

content = re.sub(
    r"    @staticmethod\s+def _compute_batchwise_differences.*?return \(-2 \* a_times_b \+ a_times_a \+ b_times_b\)\.clamp\(0, None\)\.sqrt\(\)",
    new_func,
    content,
    flags=re.DOTALL
)

with open("patchcore-inspection/src/patchcore/sampler.py", "w") as f:
    f.write(content)

