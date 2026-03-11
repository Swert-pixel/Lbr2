import numpy as np
import time
import platform
import psutil
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
import io

# Константы
MATRIX_SIZE = 512
N = MATRIX_SIZE
FLOPS_COUNT = 2 * N ** 3  # теоретическое количество операций


def generate_matrices():
    """Генерация двух случайных матриц размера N x N с элементами double"""
    np.random.seed(42)  # для воспроизводимости
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    return A, B


def measure_time(func, *args, **kwargs):
    """Измерение времени выполнения функции в секундах"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def calculate_performance(execution_time):
    """Расчет производительности в MFLOPS"""
    if execution_time > 0:
        mflops = (FLOPS_COUNT / execution_time) * 1e-6
        return mflops
    return 0


# Вариант 1: Классический алгоритм из линейной алгебры (тройной вложенный цикл)
def classic_matrix_multiply(A, B):
    """Перемножение матриц по формуле: C[i][j] = sum(A[i][k] * B[k][j])"""
    n = len(A)
    C = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


# Вариант 2: BLAS через NumPy (использует оптимизированные библиотеки)
def blas_matrix_multiply(A, B):
    """Использование оптимизированной реализации BLAS через NumPy"""
    return np.dot(A, B)  # или A @ B


#Вариант 3
def optimized_matrix_multiply(A, B):

    n = len(A)
    block_size = 256  # Оптимальный размер блока для матрицы 1024

    C = np.zeros((n, n), dtype=np.float64)

    # Основной цикл по блокам
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)

            # Временная матрица для накопления результата блока
            # Используем float64 для максимальной точности
            block_C = np.zeros((i_end - i, j_end - j), dtype=np.float64)

            for k in range(0, n, block_size):
                k_end = min(k + block_size, n)

                # Умножение блоков с использованием векторизованных операций
                # np.dot вызывает оптимизированный BLAS под капотом
                block_C += np.dot(
                    A[i:i_end, k:k_end],
                    B[k:k_end, j:j_end]
                )

            # Сохраняем результат блока
            C[i:i_end, j:j_end] = block_C

    return C



def get_system_info():
    """Получение информации о системе и используемой BLAS-библиотеке"""
    info = {
        "Процессор": platform.processor(),
        "Ядра (логические)": psutil.cpu_count(logical=True),
        "Ядра (физические)": psutil.cpu_count(logical=False),
        "Оперативная память (ГБ)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Python": platform.python_version(),
        "NumPy": np.__version__
    }

    # Получение информации о BLAS
    blas_info_str = "Информация о BLAS не найдена"
    try:
        # Захватываем вывод show_config() в переменную
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        np.show_config()
        sys.stdout = old_stdout
        full_config = new_stdout.getvalue()

        # Ищем ключевые слова
        if 'mkl' in full_config.lower():
            blas_info_str = "Intel MKL"
        elif 'openblas' in full_config.lower():
            blas_info_str = "OpenBLAS"
        elif 'blis' in full_config.lower():
            blas_info_str = "BLIS"
        elif 'accelerate' in full_config.lower():
            blas_info_str = "Accelerate (macOS)"
        else:
            # Сохраняем первые 200 символов конфига
            blas_info_str = full_config[:200].replace('\n', ' ') + '...'
    except Exception as e:
        blas_info_str = f"Не удалось получить информацию: {e}"

    info["BLAS"] = blas_info_str
    return info


def run_benchmark():
    """Запуск всех тестов и сбор результатов"""

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ УМНОЖЕНИЯ МАТРИЦ")
    print(f"Размер матриц: {N} x {N}")
    print(f"Теоретическое количество операций: {FLOPS_COUNT:.2e}")
    print("=" * 60)

    # Информация о системе
    print("\nПолучение информации о системе...")
    sys_info = get_system_info()
    print("\nИнформация о системе:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")

    # Генерация матриц
    print("\nГенерация матриц...")
    A, B = generate_matrices()
    print(f"  Матрицы размером {N}x{N} сгенерированы")

    # Проверка корректности результатов
    print("\nПроверка корректности результатов...")
    reference = np.dot(A, B)  # эталонный результат

    results = []

    # Тестирование вариантов
    variants = [
        ("Классический (тройной цикл)", classic_matrix_multiply),
        ("BLAS (NumPy dot)", blas_matrix_multiply),
        ("Оптимизированный (блочный, блок 256)", optimized_matrix_multiply),
    ]

    for idx, (name, func) in enumerate(variants):
        print(f"\n{'-' * 40}")
        print(f"Вариант {idx + 1}: {name}")
        print(f"  Запуск...")

        try:
            # Измерение времени
            C, exec_time = measure_time(func, A, B)

            # Проверка корректности (с меньшей точностью для классического алгоритма)
            if idx == 0:  # Классический алгоритм
                if not np.allclose(C, reference, rtol=1e-5, atol=1e-5):
                    print(f"  ⚠️  Результат классического алгоритма отличается от эталона")
            else:
                if not np.allclose(C, reference, rtol=1e-8, atol=1e-8):
                    print(f"  ⚠️  Результат может отличаться от эталона")

            # Расчет производительности
            mflops = calculate_performance(exec_time)

            result_entry = {
                "Вариант": name,
                "Время (с)": exec_time,
                "MFLOPS": mflops
            }
            results.append(result_entry)

            print(f"  ✅ Время выполнения: {exec_time:.4f} с")
            print(f"  ✅ Производительность: {mflops:.2f} MFLOPS")

        except Exception as e:
            print(f"  ❌ ОШИБКА при выполнении: {e}")
            import traceback
            traceback.print_exc()

    return results


def plot_results(results):
    """Построение графиков результатов"""
    if not results:
        print("Нет результатов для построения графиков")
        return

    names = [r["Вариант"] for r in results]
    times = [r["Время (с)"] for r in results]
    mflops = [r["MFLOPS"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График времени
    bars1 = ax1.bar(names, times, color=['red', 'green', 'blue'])
    ax1.set_xlabel('Алгоритм')
    ax1.set_ylabel('Время (секунды)')
    ax1.set_title('Сравнение времени выполнения')
    ax1.set_yscale('log')
    ax1.tick_params(axis='x', rotation=15)
    for bar, t in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{t:.3f} с', ha='center', va='bottom', fontsize=9, rotation=0)

    # График производительности
    bars2 = ax2.bar(names, mflops, color=['red', 'green', 'blue'])
    ax2.set_xlabel('Алгоритм')
    ax2.set_ylabel('Производительность (MFLOPS)')
    ax2.set_title('Сравнение производительности')
    ax2.tick_params(axis='x', rotation=15)
    for bar, m in zip(bars2, mflops):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{m:.0f} MFLOPS', ha='center', va='bottom', fontsize=9, rotation=0)

    plt.tight_layout()

    # Сохраняем график
    try:
        plt.savefig('matrix_multiplication_benchmark.png', dpi=150, bbox_inches='tight')
        print("\nГрафики сохранены в 'matrix_multiplication_benchmark.png'")
    except Exception as e:
        print(f"Не удалось сохранить график: {e}")

    # Показываем график
    plt.show()


def main():
    """Основная функция"""

    print("Драгель Максим Вячеславович")
    print("Группа: 09.03.01-ПОВа-о25")
    print("Запуск программы тестирования умножения матриц")
    print("\n" + "=" * 60)
    print("=" * 60)

    results = run_benchmark()

    if not results:
        print("\n❌ Не удалось получить результаты тестирования")
        return

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)

    # Подготовка данных для таблицы
    headers = ["Вариант", "Время (с)", "MFLOPS", "Относительно BLAS"]
    table_data = []

    blas_mflops = None
    for r in results:
        if "BLAS" in r["Вариант"]:
            blas_mflops = r["MFLOPS"]

    for r in results:
        if blas_mflops and blas_mflops > 0:
            if "BLAS" in r["Вариант"]:
                ratio_str = "100%"
            else:
                ratio = (r["MFLOPS"] / blas_mflops * 100)
                ratio_str = f"{ratio:.1f}%"
        else:
            ratio_str = "N/A"

        table_data.append([
            r["Вариант"],
            f"{r['Время (с)']:.4f}",
            f"{r['MFLOPS']:.2f}",
            ratio_str
        ])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Анализ результатов
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    if len(results) >= 3 and blas_mflops:
        classic_mflops = results[0]["MFLOPS"]
        blas_mflops_val = blas_mflops
        opt_mflops = results[2]["MFLOPS"]

        classic_ratio = (classic_mflops / blas_mflops_val) * 100
        opt_ratio = (opt_mflops / blas_mflops_val) * 100

        print(f"\n📊 BLAS производительность: {blas_mflops_val:.2f} MFLOPS (100%)")
        print(f"📊 Классический алгоритм: {classic_mflops:.2f} MFLOPS ({classic_ratio:.2f}%)")
        print(f"📊 Оптимизированный алгоритм: {opt_mflops:.2f} MFLOPS ({opt_ratio:.2f}%)")

        if opt_ratio >= 30:
            print(
                f"\n✅ ТРЕБОВАНИЕ ВЫПОЛНЕНО: оптимизированный алгоритм достиг {opt_ratio:.1f}% от производительности BLAS (требуется ≥30%)")
            print(f"   Отлично! Ваш алгоритм работает достаточно быстро.")
        else:
            print(f"\n❌ ТРЕБОВАНИЕ НЕ ВЫПОЛНЕНО: оптимизированный алгоритм достиг только {opt_ratio:.1f}% от BLAS")
            print(f"   Нужно ≥30%. Попробуйте уменьшить размер блока до 128 или увеличить до 512.")


    # Ускорение классического алгоритма
    if len(results) >= 2:
        speedup_blas = results[0]["Время (с)"] / results[1]["Время (с)"]
        speedup_opt = results[0]["Время (с)"] / results[2]["Время (с)"]
        print(f"📊 BLAS быстрее классического в {speedup_blas:.0f} раз")
        print(f"📊 Оптимизированный быстрее классического в {speedup_opt:.0f} раз")

    # Построение графиков
    try:
        plot_results(results)
    except Exception as e:
        print(f"⚠️ Не удалось построить графики: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nПрограмма завершена")