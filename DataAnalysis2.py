import pickle

def extract_params(params, result=None):
    if result is None:
        result = []
    if isinstance(params, dict):
        for v in params.values():
            extract_params(v, result)
    elif isinstance(params, (list, tuple)):
        for v in params:
            extract_params(v, result)
    elif hasattr(params, 'tolist'):  # numpy 배열 등
        for v in params.tolist():
            extract_params(v, result)
    elif isinstance(params, (int, float)):
        result.append(params)
    return result

def to_custom_8bit(val):
    # 정수 2자리, 소수 6자리
    abs_val = abs(val)
    int_part = int(abs_val)
    frac_part = abs_val - int_part
    int_bin = format(int_part, '02b')[-2:]  # 2비트 정수
    frac_bin = ''
    f = frac_part
    for _ in range(6):  # 6비트 소수
        f *= 2
        bit = int(f)
        frac_bin += str(bit)
        f -= bit
    bin_str = int_bin + '.' + frac_bin
    # 소수점 제거 후 8비트로 만듦
    bin8 = (int_bin + frac_bin)[:8]
    return bin8

with open('mlp_MNIST_weight.pkl', 'rb') as f:
    data = pickle.load(f)

params_list = extract_params(data)

total = 0
count_1111 = 0
count_0000 = 0

for param in params_list:
    bin8 = to_custom_8bit(param)
    total += 1
    if bin8.startswith('1111'):
        count_1111 += 1
    elif bin8.startswith('0000'):
        count_0000 += 1

ratio_1111 = count_1111 / total if total else 0
ratio_0000 = count_0000 / total if total else 0

print(f"전체 파라미터 수: {total}")
print(f"맨 앞 4자리가 1111인 비율: {ratio_1111:.4f}")
print(f"맨 앞 4자리가 0000인 비율: {ratio_0000:.4f}")