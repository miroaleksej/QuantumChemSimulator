# QuantumChemSimulator: Квантово-химический симулятор с гибридными алгоритмами

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

**QuantumChemSimulator** - это научно-ориентированный программный комплекс для моделирования электронной структуры молекул с использованием квантовых и классических вычислительных методов. Проект объединяет традиционные методы квантовой химии с современными квантовыми алгоритмами для точного расчета молекулярных свойств.

![Молекулярная визуализация](https://via.placeholder.com/800x400?text=3D+Визуализация+Молекул)

## 🔍 Основные возможности

- **Гибридные методы вычислений**:
  - Хартри-Фок (HF)
  - Функционалы плотности (DFT: B3LYP, PBE)
  - Вариационный квантовый эйгенсолвер (VQE)
  - Гибридный метод DFT+VQE
  - Пост-HF методы (MP2, CCSD)

- **Квантовые алгоритмы**:
  - Анзацы UCCSD и аппаратно-эффективные
  - Преобразование гамильтониана (Джордана-Вигнера)
  - Реалистичные модели квантовых шумов

- **Анализ и визуализация**:
  - 3D визуализация молекулярных орбиталей
  - Анализ зависимости от базисного набора
  - Оптимизация геометрии молекул
  - Сравнение с экспериментальными данными

## 🚀 Обновление через update.py

Файл `update.py` добавляет в симулятор расширенные научные функции и улучшения:

```bash
python update.py
```

### 📦 Что добавляет обновление:

1. **Улучшенные базисные функции**:
   - Реализация контрактированных гауссовых функций
   - Поддержка высших угловых моментов (p, d, f-орбитали)

2. **Точный расчет интегралов**:
   - Алгоритм Оохата-Кога для интегралов перекрывания
   - Алгоритм Мак-Мюрчи для двуэлектронных интегралов

3. **Ускорение сходимости SCF**:
   - DIIS-экстраполяция матрицы Фока
   ```python
   diis = DIIS(max_diis=6)
   F = diis.extrapolate(F, error)
   ```

4. **Квантовые преобразования**:
   - Полное преобразование гамильтониана в операторы Паули
   ```python
   converter = HamiltonianConverter(hf_result)
   qubit_hamiltonian = converter.get_qubit_hamiltonian()
   ```

5. **Реалистичные анзацы**:
   - Генерация квантовых схем UCCSD
   ```python
   uccsd = UCCSDAnsatz(num_qubits, num_electrons)
   circuit = uccsd.build_circuit()
   ```

6. **Расширенная поддержка DFT**:
   - Реализация функционалов LDA, PBE, B3LYP, wB97X
   ```python
   dft = DFTModule(molecule, functional='B3LYP')
   energy = dft.calculate_energy()
   ```

7. **Физические шумовые модели**:
   - Деполяризующий шум
   - Амплитудное затухание
   - Тепловая релаксация
   ```python
   noise_model = QuantumNoiseModel(T1=50e-6, T2=70e-6)
   noisy_circuit = noise_model.apply_depolarizing_error(circuit, qubit, 0.01)
   ```

8. **Производительность**:
   - GPU-ускорение через CuPy
   - Работа с разреженными матрицами
   ```python
   solver = GPUSolver()
   eigenvalues = solver.solve_eigen(matrix)
   ```

9. **Расширенные методы**:
   - MP2 и CCSD для точных расчетов
   ```python
   post_hf = PostHFMethods(hf_result)
   mp2_energy = post_hf.mp2_energy()
   ```

10. **3D визуализация**:
    - Интерактивное отображение молекулярных орбиталей
    ```python
    visualizer = OrbitalVisualizer(molecule, mo_coeff, basis_set)
    visualizer.plot_orbital(0)
    ```

11. **Интеграция с PySCF**:
    ```python
    pyscf_mol = PySCFInterface.to_pyscf(molecule)
    ```

12. **Оптимизация геометрии**:
    ```python
    optimizer = GeometryOptimizer(simulator)
    result = optimizer.optimize()
    ```

## ⚙️ Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/QuantumChemSimulator.git
cd QuantumChemSimulator

# Установка зависимостей
pip install -r requirements.txt

# Применение обновления
python update.py

# Запуск тестов
python -m unittest discover
```

## 📚 Примеры использования

### Расчет энергии молекулы воды
```python
from QuantumChemSimulator import QuantumChemSimulator

simulator = QuantumChemSimulator(basis_set='cc-pVDZ')
simulator.load_molecule("O 0 0 0; H 0 0.76 -0.5; H 0 -0.76 -0.5")

# Расчет гибридной энергии
energy = simulator.calculate_energy('Hybrid', gamma=0.55)
print(f"Гибридная энергия: {energy:.6f} Ha")

# Визуализация HOMO орбитали
simulator.visualize_orbital(orbital_index=5)
```

### Оптимизация геометрии
```python
optimizer = simulator.geometry_optimizer(method='VQE')
result = optimizer.optimize(max_iter=50)
optimizer.plot_convergence()
```

### Запуск из командной строки
```bash
python simulate.py --molecule "N 0 0 0; N 0 0 1.1" \
                  --method vqe \
                  --ansatz uccsd \
                  --basis cc-pVTZ \
                  --max_iter 100
```

## 📊 Сравнение с аналогами

| Функция | QuantumChemSimulator | PySCF | Qiskit Nature |
|---------|----------------------|-------|---------------|
| Гибридные алгоритмы | ✅ | ❌ | ⚠️ |
| 3D визуализация | ✅ | ❌ | ❌ |
| Шумовые модели | ✅ | ❌ | ✅ |
| GPU ускорение | ✅ | ✅ | ⚠️ |
| Поддержка VQE | ✅ | ❌ | ✅ |
| Гибрид DFT+VQE | ✅ | ❌ | ❌ |
| Обновление через update.py | ✅ | ❌ | ❌ |

## 📜 Лицензия

Проект распространяется под лицензией MIT. Полный текст лицензии доступен в файле [LICENSE](LICENSE).

## 🤝 Как внести вклад

1. Форкните репозиторий
2. Создайте ветку для вашей функции (`git checkout -b feature/AmazingFeature`)
3. Зафиксируйте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Запушьте ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## ✉️ Контакты
