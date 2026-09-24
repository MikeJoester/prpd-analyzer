import re

with open("patchcore-inspection/src/patchcore/sampler.py", "r") as f:
    content = f.read()

new_func = """    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        b_times_b = (matrix_b ** 2).sum(dim=1).unsqueeze(0)
        
        # Process in chunks to avoid OOM
        chunk_size = 1000000
        results = []
        for i in range(0, matrix_a.size(0), chunk_size):
            chunk_a = matrix_a[i:i+chunk_size]
            a_times_a = (chunk_a ** 2).sum(dim=1, keepdim=True)
            a_times_b = chunk_a.mm(matrix_b.T)
            res = (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()
            results.append(res)
            del chunk_a, a_times_a, a_times_b, res
            
        return torch.cat(results, dim=0)"""

content = re.sub(
    r"    @staticmethod\s+def _compute_batchwise_differences.*?return \(-2 \* a_times_b \+ a_times_a \+ b_times_b\)\.clamp\(0, None\)\.sqrt\(\)",
    new_func,
    content,
    flags=re.DOTALL
)

with open("patchcore-inspection/src/patchcore/sampler.py", "w") as f:
    f.write(content)

