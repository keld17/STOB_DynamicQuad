import pickle
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

def float_to_fixed_point(num, integer_bits=2, fractional_bits=6):
    """
    하나의 실수를 2의 보수 고정소수점 이진수 문자열로 변환합니다.
    """
    total_bits = integer_bits + fractional_bits
    min_val = - (2 ** (integer_bits - 1))
    max_val = (2 ** (integer_bits - 1) - 1) + (1 - 2 ** -fractional_bits)

    if num >= max_val:
        num = max_val
    elif num < min_val:
        num = min_val

    scaled_value = int(round(num * (2 ** fractional_bits)))

    if scaled_value < 0:
        scaled_value = scaled_value & ((1 << total_bits) - 1)

    binary_representation = bin(scaled_value)[2:]
    padded_binary = binary_representation.zfill(total_bits)
    fixed_point_string = f"{padded_binary[:integer_bits]}.{padded_binary[integer_bits:]}"

    return fixed_point_string

def analyze_binary_patterns(data, integer_bits=2, fractional_bits=6):
    """
    데이터를 이진수로 변환한 뒤, 특정 패턴으로 시작하는 값들의 통계를 분석합니다.
    """
    def _get_all_numbers(sub_data):
        """데이터 구조를 순회하며 모든 숫자를 리스트로 반환하는 내부 함수"""
        numbers = []
        if isinstance(sub_data, (list, tuple)):
            for item in sub_data:
                numbers.extend(_get_all_numbers(item))
        elif isinstance(sub_data, dict):
            for value in sub_data.values():
                numbers.extend(_get_all_numbers(value))
        elif isinstance(sub_data, np.ndarray):
            numbers.extend(sub_data.flatten().tolist())
        elif isinstance(sub_data, (int, float, np.number)):
            numbers.append(float(sub_data))
        return numbers

    all_numbers = _get_all_numbers(data)
    total_count = len(all_numbers)

    if total_count == 0:
        print("분석할 데이터가 없습니다.")
        return

    binary_strings = [
        float_to_fixed_point(num, integer_bits, fractional_bits) for num in all_numbers
    ]

    count_1111 = 0
    count_0000 = 0
    for b_string in binary_strings:
        # 소수점을 제거하여 순수 이진수 문자열로 만듦
        raw_binary = b_string.replace('.', '')
        
        if raw_binary.startswith('1111'):
            count_1111 += 1
        elif raw_binary.startswith('0000'):
            count_0000 += 1
    
    total_pattern_count = count_1111 + count_0000
    percentage = (total_pattern_count / total_count) * 100

    print("=" * 40)
    print("이진수 패턴 분석 결과")
    print("-" * 40)
    print(f" - 전체 데이터 개수: {total_count}개")
    print(f" - '1111' 시작 개수: {count_1111}개")
    print(f" - '0000' 시작 개수: {count_0000}개")
    print(f" -> 총 {total_pattern_count}개가 해당 패턴으로 시작합니다.")
    print(f" -> 이는 전체 데이터의 약 {percentage:.2f}%를 차지합니다.")
    print("=" * 40)

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    file_name = 'mlp_MNIST_weight.pkl'

    try:
        with open(file_name, 'rb') as f:
            weights_data = pickle.load(f, encoding='latin1')

        print(f"'{file_name}' 파일 로드 성공!")
        
        # 수정된 분석 함수 호출
        analyze_binary_patterns(weights_data, integer_bits=2, fractional_bits=6)

    except FileNotFoundError:
        print(f"오류: '{file_name}' 파일을 찾을 수 없습니다. 파일이 올바른 위치에 있는지 확인해주세요.")
    except Exception as e:
        print(f"파일을 처리하는 중 오류가 발생했습니다: {e}")