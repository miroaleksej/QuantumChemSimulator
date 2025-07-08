import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Optional
from collections import defaultdict
import warnings

# Отключение предупреждений для чистоты вывода
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --------------------------
# Базовые научные константы
# --------------------------
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
EV_TO_KJ_MOL = 96.485

class QuantumChemistryError(Exception):
    """Специализированное исключение для ошибок квантовой химии"""
    pass

# --------------------------
# Модуль молекулярной геометрии
# --------------------------
class Atom:
    """Представление атома с его свойствами"""
    ATOMIC_NUMBERS = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20
    }
    
    COVALENT_RADII = {
        1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71,
        8: 0.66, 9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11,
        15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76
    }
    
    def __init__(self, symbol: str, x: float, y: float, z: float):
        self.symbol = symbol.capitalize()
        self.position = np.array([x, y, z], dtype=float)
        
        if symbol not in self.ATOMIC_NUMBERS:
            raise QuantumChemistryError(f"Неизвестный атом: {symbol}")
        
        self.atomic_number = self.ATOMIC_NUMBERS[symbol]
        self.covalent_radius = self.COVALENT_RADII[self.atomic_number]
    
    def distance_to(self, other: 'Atom') -> float:
        """Расчет расстояния до другого атома"""
        return np.linalg.norm(self.position - other.position)
    
    def __repr__(self):
        return f"{self.symbol}({self.position[0]:.4f}, {self.position[1]:.4f}, {self.position[2]:.4f})"

class Molecule:
    """Представление молекулы как набора атомов"""
    def __init__(self, atoms: List[Atom], charge: int = 0, spin: int = 0):
        self.atoms = atoms
        self.charge = charge
        self.spin = spin
        self._bonds = None
    
    @classmethod
    def from_string(cls, mol_str: str, charge: int = 0, spin: int = 0) -> 'Molecule':
        """Создание молекулы из строки формата 'H 0 0 0; O 0 0 1.0'"""
        atoms = []
        for atom_str in mol_str.split(';'):
            parts = atom_str.strip().split()
            if len(parts) < 4:
                continue
            symbol = parts[0]
            coords = [float(x) for x in parts[1:4]]
            atoms.append(Atom(symbol, *coords))
        return cls(atoms, charge, spin)
    
    def get_bonds(self) -> List[Tuple[int, int]]:
        """Определение связей между атомами на основе ковалентных радиусов"""
        if self._bonds is not None:
            return self._bonds
        
        bonds = []
        n_atoms = len(self.atoms)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = self.atoms[i].distance_to(self.atoms[j])
                max_dist = 1.3 * (self.atoms[i].covalent_radius + self.atoms[j].covalent_radius)
                
                if dist < max_dist:
                    bonds.append((i, j))
        
        self._bonds = bonds
        return bonds
    
    def get_center_of_mass(self) -> np.ndarray:
        """Расчет центра масс молекулы"""
        total_mass = 0.0
        com = np.zeros(3)
        
        for atom in self.atoms:
            mass = atom.atomic_number  # Упрощение: используем атомный номер как массу
            com += mass * atom.position
            total_mass += mass
        
        return com / total_mass
    
    def translate(self, vector: np.ndarray):
        """Перенос молекулы на заданный вектор"""
        for atom in self.atoms:
            atom.position += vector
    
    def center(self):
        """Центрирование молекулы в начале координат"""
        com = self.get_center_of_mass()
        self.translate(-com)
    
    def __repr__(self):
        atom_str = ";\n".join(str(atom) for atom in self.atoms)
        return f"Molecule(charge={self.charge}, spin={self.charge}):\n{atom_str}"

# --------------------------
# Модуль базисных функций
# --------------------------
class BasisFunction(ABC):
    """Абстрактный класс базисной функции"""
    @abstractmethod
    def evaluate(self, r: np.ndarray) -> float:
        pass
    
    @property
    @abstractmethod
    def center(self) -> np.ndarray:
        pass

class GaussianBasis(BasisFunction):
    """Гауссова базисная функция"""
    def __init__(self, center: np.ndarray, exponent: float, coefficients: List[float], 
                 powers: List[int], normalized: bool = True):
        self._center = np.array(center)
        self.exponent = exponent
        self.coefficients = np.array(coefficients)
        self.powers = np.array(powers)
        self.normalized = normalized
        
        if normalized:
            self.normalize()
    
    def normalize(self):
        """Нормализация гауссовой функции"""
        norm = 0.0
        for c1, p1 in zip(self.coefficients, self.powers):
            for c2, p2 in zip(self.coefficients, self.powers):
                # Упрощенный расчет нормы
                total_power = sum(p1 + p2)
                norm += c1 * c2 * (np.pi / (2 * self.exponent)) ** (3/2) * (4 * self.exponent) ** (total_power / 2)
        
        norm_factor = 1.0 / np.sqrt(norm)
        self.coefficients *= norm_factor
    
    @property
    def center(self) -> np.ndarray:
        return self._center
    
    def evaluate(self, r: np.ndarray) -> float:
        """Вычисление значения функции в точке"""
        dr = r - self.center
        r2 = np.dot(dr, dr)
        
        value = 0.0
        for coeff, powers in zip(self.coefficients, self.powers):
            # Полиномиальная часть
            poly = 1.0
            for i, power in enumerate(powers):
                poly *= dr[i] ** power
            
            # Экспоненциальная часть
            exp_part = np.exp(-self.exponent * r2)
            
            value += coeff * poly * exp_part
        
        return value

class BasisSet:
    """Набор базисных функций для молекулы"""
    STO_3G = {
        'H': [GaussianBasis([0,0,0], 3.42525091, [0.15432897], [0,0,0]),
        'C': [GaussianBasis([0,0,0], 2.94124940, [-0.09996723], [0,0,0]),
        'N': [GaussianBasis([0,0,0], 3.78045588, [-0.09996723], [0,0,0]),
        'O': [GaussianBasis([0,0,0], 5.03315130, [-0.09996723], [0,0,0]),
        # Упрощенный вариант для демонстрации
    }
    
    def __init__(self, name: str = 'STO-3G'):
        self.name = name
        self.basis_functions = defaultdict(list)
        
        if name in ['STO-3G', 'sto-3g']:
            self.basis_functions = self.STO_3G
        else:
            raise NotImplementedError(f"Базисный набор {name} не реализован")
    
    def get_functions_for_atom(self, atom: Atom) -> List[BasisFunction]:
        """Получение базисных функций для атома"""
        functions = self.basis_functions.get(atom.symbol, [])
        
        # Центрируем функции на положении атома
        centered_functions = []
        for func in functions:
            # Создаем новую функцию с центром в положении атома
            new_func = GaussianBasis(
                center=atom.position,
                exponent=func.exponent,
                coefficients=func.coefficients,
                powers=func.powers,
                normalized=func.normalized
            )
            centered_functions.append(new_func)
        
        return centered_functions

# --------------------------
# Модуль интегралов
# --------------------------
class IntegralCalculator:
    """Калькулятор молекулярных интегралов"""
    def __init__(self, molecule: Molecule, basis_set: BasisSet):
        self.molecule = molecule
        self.basis_set = basis_set
        self._basis_functions = self._prepare_basis_functions()
        self._overlap_matrix = None
        self._kinetic_matrix = None
        self._nuclear_matrix = None
        self._eri_tensor = None
    
    def _prepare_basis_functions(self) -> List[BasisFunction]:
        """Подготовка списка всех базисных функций"""
        basis_functions = []
        for atom in self.molecule.atoms:
            atom_funcs = self.basis_set.get_functions_for_atom(atom)
            basis_functions.extend(atom_funcs)
        return basis_functions
    
    def calculate_overlap_matrix(self) -> np.ndarray:
        """Расчет матрицы перекрывания"""
        n = len(self._basis_functions)
        S = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Упрощенный расчет перекрывания
                # В реальной реализации используется численное интегрирование
                dx = np.linalg.norm(self._basis_functions[i].center - 
                                   self._basis_functions[j].center)
                exponent_i = self._basis_functions[i].exponent
                exponent_j = self._basis_functions[j].exponent
                
                # Приближенное значение перекрывания
                overlap = np.exp(-0.5 * (exponent_i + exponent_j) * dx**2)
                S[i, j] = S[j, i] = overlap
        
        self._overlap_matrix = S
        return S
    
    def calculate_kinetic_matrix(self) -> np.ndarray:
        """Расчет матрицы кинетической энергии"""
        n = len(self._basis_functions)
        T = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Упрощенный расчет кинетической энергии
                dx = np.linalg.norm(self._basis_functions[i].center - 
                                   self._basis_functions[j].center)
                exponent_i = self._basis_functions[i].exponent
                exponent_j = self._basis_functions[j].exponent
                
                # Приближенное значение
                kinetic = exponent_i * exponent_j * dx**2 * np.exp(-0.3 * (exponent_i + exponent_j) * dx**2)
                T[i, j] = T[j, i] = kinetic
        
        self._kinetic_matrix = T
        return T
    
    def calculate_nuclear_matrix(self) -> np.ndarray:
        """Расчет матрицы ядерного притяжения"""
        n = len(self._basis_functions)
        V = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Упрощенный расчет ядерного притяжения
                value = 0.0
                for atom in self.molecule.atoms:
                    dx_i = np.linalg.norm(self._basis_functions[i].center - atom.position)
                    dx_j = np.linalg.norm(self._basis_functions[j].center - atom.position)
                    
                    exponent_i = self._basis_functions[i].exponent
                    exponent_j = self._basis_functions[j].exponent
                    
                    # Приближенное значение
                    term = -atom.atomic_number * np.exp(-0.2 * (exponent_i + exponent_j) * (dx_i + dx_j))
                    value += term
                
                V[i, j] = V[j, i] = value
        
        self._nuclear_matrix = V
        return V
    
    def calculate_electron_repulsion(self) -> np.ndarray:
        """Расчет тензора электронного отталкивания (ERI)"""
        n = len(self._basis_functions)
        eri = np.zeros((n, n, n, n))
        
        # Упрощенная реализация - в реальности требует сложных вычислений
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        # Простое приближение
                        d_ij = np.linalg.norm(self._basis_functions[i].center - 
                                             self._basis_functions[j].center)
                        d_kl = np.linalg.norm(self._basis_functions[k].center - 
                                             self._basis_functions[l].center)
                        
                        eri[i, j, k, l] = 1.0 / (1.0 + d_ij + d_kl)
        
        self._eri_tensor = eri
        return eri
    
    def compress_integrals(self, threshold: float = 1e-5) -> Dict[str, np.ndarray]:
        """Сжатие интегралов с учетом симметрии"""
        n = len(self._basis_functions)
        
        # Матрица перекрывания
        S_compressed = self._overlap_matrix.copy()
        S_compressed[np.abs(S_compressed) < threshold] = 0.0
        
        # Матрица кинетической энергии
        T_compressed = self._kinetic_matrix.copy()
        T_compressed[np.abs(T_compressed) < threshold] = 0.0
        
        # Матрица ядерного притяжения
        V_compressed = self._nuclear_matrix.copy()
        V_compressed[np.abs(V_compressed) < threshold] = 0.0
        
        # Тензор ERI
        eri_compressed = self._eri_tensor.copy()
        eri_compressed[np.abs(eri_compressed) < threshold] = 0.0
        
        return {
            'overlap': S_compressed,
            'kinetic': T_compressed,
            'nuclear': V_compressed,
            'eri': eri_compressed
        }

# --------------------------
# Модуль метода Хартри-Фока
# --------------------------
class HartreeFock:
    """Реализация метода Хартри-Фока"""
    def __init__(self, molecule: Molecule, basis_set: BasisSet):
        self.molecule = molecule
        self.basis_set = basis_set
        self.integral_calculator = IntegralCalculator(molecule, basis_set)
        self.integrals = None
        self.density_matrix = None
        self.fock_matrix = None
        self.mo_coeffs = None
        self.mo_energies = None
        self.energy = None
        self.converged = False
    
    def initialize(self):
        """Инициализация расчета"""
        # Расчет всех интегралов
        self.integral_calculator.calculate_overlap_matrix()
        self.integral_calculator.calculate_kinetic_matrix()
        self.integral_calculator.calculate_nuclear_matrix()
        self.integral_calculator.calculate_electron_repulsion()
        
        # Получение интегралов
        self.integrals = self.integral_calculator.compress_integrals()
        
        # Начальная матрица плотности (нулевая или из геометрии)
        n_basis = len(self.integral_calculator._basis_functions)
        self.density_matrix = np.zeros((n_basis, n_basis))
    
    def build_fock_matrix(self) -> np.ndarray:
        """Построение матрицы Фока"""
        S = self.integrals['overlap']
        T = self.integrals['kinetic']
        V = self.integrals['nuclear']
        eri = self.integrals['eri']
        P = self.density_matrix
        
        # Кинетическая энергия + ядерное притяжение
        H_core = T + V
        
        # Двухэлектронная часть
        G = np.zeros_like(H_core)
        n = H_core.shape[0]
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        # Формула для матрицы Фока
                        coulomb = eri[i, j, k, l]
                        exchange = eri[i, l, k, j]
                        G[i, j] += P[k, l] * (coulomb - 0.5 * exchange)
        
        self.fock_matrix = H_core + G
        return self.fock_matrix
    
    def solve(self, max_iter: int = 50, tol: float = 1e-6) -> float:
        """Решение уравнений Хартри-Фока"""
        self.initialize()
        S = self.integrals['overlap']
        
        # Диагонализация матрицы перекрывания
        eigvals, eigvecs = np.linalg.eigh(S)
        # Удаление линейных зависимостей
        idx = eigvals > 1e-8
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Матрица X для ортогонализации
        X = eigvecs.dot(np.diag(1.0 / np.sqrt(eigvals)))
        
        prev_energy = 0.0
        n_electrons = sum(atom.atomic_number for atom in self.molecule.atoms) - self.molecule.charge
        
        for iteration in range(max_iter):
            # Построение матрицы Фока
            F = self.build_fock_matrix()
            
            # Ортогонализация
            F_ortho = X.T.dot(F).dot(X)
            
            # Диагонализация
            mo_energies, C_ortho = np.linalg.eigh(F_ortho)
            self.mo_coeffs = X.dot(C_ortho)
            
            # Обновление матрицы плотности
            P_new = np.zeros_like(self.density_matrix)
            for i in range(n_electrons // 2):
                mo_i = self.mo_coeffs[:, i]
                P_new += 2 * np.outer(mo_i, mo_i)
            
            # Расчет полной энергии
            self.energy = 0.5 * np.sum(P_new * (self.integrals['kinetic'] + self.integrals['nuclear'] + F))
            
            # Проверка сходимости
            delta_energy = abs(self.energy - prev_energy)
            delta_density = np.linalg.norm(P_new - self.density_matrix)
            
            if delta_energy < tol and delta_density < tol:
                self.converged = True
                break
            
            # Обновление для следующей итерации
            self.density_matrix = P_new
            prev_energy = self.energy
        
        self.mo_energies = mo_energies
        return self.energy
    
    def get_molecular_orbitals(self) -> np.ndarray:
        """Получение молекулярных орбиталей"""
        return self.mo_coeffs
    
    def get_orbital_energies(self) -> np.ndarray:
        """Получение энергий орбиталей"""
        return self.mo_energies

# --------------------------
# Модуль VQE
# --------------------------
class VQE:
    """Вариационный квантовый эйгенсолвер"""
    def __init__(self, hamiltonian, ansatz, optimizer='L-BFGS-B', initial_params=None):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.initial_params = initial_params
        self.result = None
        self.history = {
            'energy': [],
            'params': [],
            'gradient': []
        }
    
    def energy_function(self, params):
        """Функция для вычисления энергии"""
        state = self.ansatz.prepare_state(params)
        energy = np.real(np.vdot(state, self.hamiltonian @ state))
        return energy
    
    def run(self, max_iter=100, tol=1e-6, callback=None):
        """Запуск оптимизации VQE"""
        from scipy.optimize import minimize
        
        if self.initial_params is None:
            self.initial_params = np.random.randn(self.ansatz.num_params)
        
        def cost_function(params):
            energy = self.energy_function(params)
            self.history['energy'].append(energy)
            self.history['params'].append(params.copy())
            if callback:
                callback(energy, params)
            return energy
        
        res = minimize(
            cost_function,
            self.initial_params,
            method=self.optimizer,
            tol=tol,
            options={'maxiter': max_iter}
        )
        
        self.result = res
        return res

class Ansatz(ABC):
    """Абстрактный класс анзаца"""
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.num_params = 0
    
    @abstractmethod
    def prepare_state(self, params: np.ndarray) -> np.ndarray:
        """Подготовка квантового состояния"""
        pass

class UCCSDAnsatz(Ansatz):
    """Анзац UCCSD (Unitary Coupled Cluster Singles and Doubles)"""
    def __init__(self, num_qubits, num_electrons):
        super().__init__(num_qubits)
        self.num_electrons = num_electrons
        self._generate_excitations()
    
    def _generate_excitations(self):
        """Генерация списка возбуждений"""
        self.singles = []
        self.doubles = []
        
        # Одноэлектронные возбуждения
        for i in range(self.num_electrons):
            for a in range(self.num_electrons, self.num_qubits):
                self.singles.append((i, a))
        
        # Двухэлектронные возбуждения
        for i in range(self.num_electrons):
            for j in range(i+1, self.num_electrons):
                for a in range(self.num_electrons, self.num_qubits):
                    for b in range(a+1, self.num_qubits):
                        self.doubles.append((i, j, a, b))
        
        self.num_params = len(self.singles) + len(self.doubles)
    
    def prepare_state(self, params: np.ndarray) -> np.ndarray:
        """Подготовка состояния методом UCCSD"""
        # Начальное состояние - детерминант Хартри-Фока
        state = np.zeros(2**self.num_qubits)
        state[int('1'*self.num_electrons + '0'*(self.num_qubits - self.num_electrons), 2)] = 1.0
        
        # Применение возбуждений
        idx = 0
        for i, a in self.singles:
            theta = params[idx]
            # Применение оператора возбуждения
            # (упрощенная реализация - в реальности нужны матричные представления)
            state = self._apply_excitation(state, i, a, theta)
            idx += 1
        
        for i, j, a, b in self.doubles:
            theta = params[idx]
            # Применение оператора возбуждения
            state = self._apply_excitation(state, i, a, theta)
            state = self._apply_excitation(state, j, b, theta)
            idx += 1
        
        return state
    
    def _apply_excitation(self, state, i, a, theta):
        """Применение оператора возбуждения (упрощенное)"""
        # В реальной реализации это было бы унитарное преобразование
        # Здесь - упрощенный вариант для демонстрации
        return state * np.exp(1j * theta)

class HardwareEfficientAnsatz(Ansatz):
    """Аппаратно-эффективный анзац"""
    def __init__(self, num_qubits, depth=3):
        super().__init__(num_qubits)
        self.depth = depth
        self.num_params = num_qubits * depth * 2  # По 2 параметра на гейт
    
    def prepare_state(self, params: np.ndarray) -> np.ndarray:
        """Подготовка состояния с помощью аппаратно-эффективного анзаца"""
        # Начальное состояние |0>^n
        state = np.zeros(2**self.num_qubits)
        state[0] = 1.0
        
        # Применение слоев
        param_idx = 0
        for _ in range(self.depth):
            # Слой вращений
            for qubit in range(self.num_qubits):
                theta = params[param_idx]
                phi = params[param_idx + 1]
                param_idx += 2
                # Применение вращений (упрощенно)
                state = self._apply_rotation(state, qubit, theta, phi)
            
            # Слой энтэнглера
            for qubit in range(0, self.num_qubits-1, 2):
                # Применение CNOT
                state = self._apply_cnot(state, qubit, qubit+1)
        
        return state
    
    def _apply_rotation(self, state, qubit, theta, phi):
        """Применение вращения (упрощенно)"""
        # В реальной реализации это было бы матричное умножение
        # Здесь - просто фазовый сдвиг для демонстрации
        return state * np.exp(1j * theta)
    
    def _apply_cnot(self, state, control, target):
        """Применение CNOT (упрощенно)"""
        # В реальной реализации это было бы матричное умножение
        # Здесь - просто перестановка амплитуд
        return state

# --------------------------
# Модуль гибридного DFT+VQE
# --------------------------
class HybridDFTVQE:
    """Гибридный метод DFT+VQE"""
    def __init__(self, molecule: Molecule, basis_set: BasisSet, gamma: float = 0.5):
        self.molecule = molecule
        self.basis_set = basis_set
        self.gamma = gamma
        self.hf = HartreeFock(molecule, basis_set)
        self.vqe = None
        self.dft_energy = None
        self.hf_energy = None
        self.vqe_energy = None
        self.hybrid_energy = None
    
    def calculate_energies(self):
        """Расчет всех компонентов энергии"""
        # Расчет энергии Хартри-Фока
        self.hf_energy = self.hf.solve()
        
        # Расчет DFT энергии (упрощенный)
        self.dft_energy = self._calculate_dft_energy()
        
        # Расчет VQE энергии
        self.vqe_energy = self._calculate_vqe_energy()
        
        # Гибридная энергия
        self.hybrid_energy = self.dft_energy + self.gamma * (self.vqe_energy - self.hf_energy)
        
        return self.hybrid_energy
    
    def _calculate_dft_energy(self) -> float:
        """Упрощенный расчет DFT энергии"""
        # В реальной реализации это был бы сложный расчет
        # Здесь используем приближение: DFT ~ HF + корреляция
        return self.hf_energy * 1.05
    
    def _calculate_vqe_energy(self) -> float:
        """Расчет VQE энергии"""
        # Определение гамильтониана во втором квантовании
        # (упрощенно - используем матрицу Фока)
        hamiltonian = self.hf.fock_matrix
        
        # Определение числа электронов
        n_electrons = sum(atom.atomic_number for atom in self.molecule.atoms) - self.molecule.charge
        
        # Создание анзаца UCCSD
        num_qubits = hamiltonian.shape[0]
        ansatz = UCCSDAnsatz(num_qubits, n_electrons)
        
        # Настройка и запуск VQE
        self.vqe = VQE(hamiltonian, ansatz)
        result = self.vqe.run()
        
        return result.fun
    
    def optimize_gamma(self, gamma_range=np.linspace(0.1, 1.0, 10), reference_energy=None):
        """Оптимизация параметра gamma"""
        best_gamma = self.gamma
        best_energy = None
        best_error = float('inf')
        
        if reference_energy is None:
            # Если нет эталонной энергии, используем VQE энергию как приближение
            reference_energy = self.vqe_energy
        
        for gamma in gamma_range:
            self.gamma = gamma
            hybrid_energy = self.calculate_energies()
            error = abs(hybrid_energy - reference_energy)
            
            if error < best_error:
                best_error = error
                best_gamma = gamma
                best_energy = hybrid_energy
        
        self.gamma = best_gamma
        self.hybrid_energy = best_energy
        return best_gamma, best_energy

# --------------------------
# Модуль шумовых моделей
# --------------------------
class NoiseModel:
    """Модель шумов для квантовых вычислений"""
    def __init__(self, gate_errors: Dict[str, float], readout_errors: Dict[str, float]):
        self.validate_errors(gate_errors, readout_errors)
        self.gate_errors = gate_errors
        self.readout_errors = readout_errors
    
    def validate_errors(self, gate_errors, readout_errors):
        """Проверка реалистичности ошибок"""
        if '1q' in gate_errors and not (0 <= gate_errors['1q'] <= 0.01):
            raise ValueError("Нереалистичная ошибка 1-кубитного гейта")
        if '2q' in gate_errors and not (0 <= gate_errors['2q'] <= 0.05):
            raise ValueError("Нереалистичная ошибка 2-кубитного гейта")
        
        for prob in readout_errors.values():
            if not (0 <= prob <= 0.1):
                raise ValueError("Нереалистичная ошибка считывания")
    
    def apply_gate_error(self, state, gate_type):
        """Применение ошибки гейта (упрощенно)"""
        error_rate = self.gate_errors.get(gate_type, 0.0)
        # Упрощенное добавление шума
        noise = np.random.randn(*state.shape) * error_rate * 0.1
        return state + noise
    
    def apply_readout_error(self, measurement):
        """Применение ошибки считывания (упрощенно)"""
        p0 = self.readout_errors.get('0', 0.0)
        p1 = self.readout_errors.get('1', 0.0)
        
        # Вероятность ошибки для каждого кубита
        for i in range(len(measurement)):
            if measurement[i] == 0 and np.random.rand() < p0:
                measurement[i] = 1
            elif measurement[i] == 1 and np.random.rand() < p1:
                measurement[i] = 0
        
        return measurement

# --------------------------
# Основной класс симулятора
# --------------------------
class QuantumChemSimulator:
    """Комплексный квантово-химический симулятор"""
    def __init__(self, basis_set='STO-3G', noise_model=None, gamma=0.5):
        self.basis_set_name = basis_set
        self.basis_set = BasisSet(basis_set)
        self.noise_model = noise_model
        self.gamma = gamma
        self.molecule = None
        self.results = {}
    
    def load_molecule(self, mol_str: str, charge: int = 0, spin: int = 0):
        """Загрузка молекулы из строки"""
        self.molecule = Molecule.from_string(mol_str, charge, spin)
        self.molecule.center()
        self.results = {}
    
    def calculate_energy(self, method: str = 'VQE', ansatz_type: str = 'UCCSD', 
                         max_iter: int = 100, tol: float = 1e-6) -> float:
        """Расчет энергии молекулы"""
        if not self.molecule:
            raise QuantumChemistryError("Молекула не загружена")
        
        if method == 'HF':
            hf = HartreeFock(self.molecule, self.basis_set)
            energy = hf.solve(max_iter, tol)
            self.results['HF'] = {
                'energy': energy,
                'converged': hf.converged
            }
            return energy
        
        elif method == 'VQE':
            # Сначала вычисляем HF для получения гамильтониана
            hf = HartreeFock(self.molecule, self.basis_set)
            hf.solve()
            
            # Создаем гамильтониан
            hamiltonian = hf.fock_matrix
            
            # Определяем число электронов
            n_electrons = sum(atom.atomic_number for atom in self.molecule.atoms) - self.molecule.charge
            
            # Создаем анзац
            num_qubits = hamiltonian.shape[0]
            if ansatz_type == 'UCCSD':
                ansatz = UCCSDAnsatz(num_qubits, n_electrons)
            elif ansatz_type == 'hardware_efficient':
                ansatz = HardwareEfficientAnsatz(num_qubits)
            else:
                raise ValueError(f"Неизвестный тип анзаца: {ansatz_type}")
            
            # Настраиваем VQE
            vqe = VQE(hamiltonian, ansatz)
            result = vqe.run(max_iter, tol)
            
            self.results['VQE'] = {
                'energy': result.fun,
                'ansatz': ansatz_type,
                'iterations': len(vqe.history['energy']),
                'history': vqe.history
            }
            return result.fun
        
        elif method == 'Hybrid':
            hybrid = HybridDFTVQE(self.molecule, self.basis_set, self.gamma)
            energy = hybrid.calculate_energies()
            self.results['Hybrid'] = {
                'energy': energy,
                'gamma': hybrid.gamma,
                'HF_energy': hybrid.hf_energy,
                'DFT_energy': hybrid.dft_energy,
                'VQE_energy': hybrid.vqe_energy
            }
            return energy
        
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def calculate_basis_dependence(self, basis_sets: List[str], method: str = 'VQE') -> Dict[str, float]:
        """Исследование зависимости от базисного набора"""
        results = {}
        original_basis = self.basis_set_name
        
        for basis in basis_sets:
            self.basis_set = BasisSet(basis)
            try:
                energy = self.calculate_energy(method)
                results[basis] = energy
            except Exception as e:
                print(f"Ошибка для базиса {basis}: {str(e)}")
                results[basis] = None
        
        # Восстановление исходного базиса
        self.basis_set = BasisSet(original_basis)
        return results
    
    def plot_convergence(self, method: str = 'VQE'):
        """Построение графика сходимости"""
        if method not in self.results or 'history' not in self.results[method]:
            raise QuantumChemistryError("Нет данных о сходимости")
        
        history = self.results[method]['history']['energy']
        plt.plot(range(len(history)), history, 'o-')
        plt.xlabel('Итерация')
        plt.ylabel('Энергия (Хартри)')
        plt.title(f'Сходимость {method} для {self.molecule}')
        plt.grid(True)
        plt.show()
    
    def optimize_gamma(self, gamma_range=np.linspace(0.1, 1.0, 10), reference_energy=None) -> Tuple[float, float]:
        """Оптимизация параметра gamma в гибридном методе"""
        hybrid = HybridDFTVQE(self.molecule, self.basis_set, self.gamma)
        return hybrid.optimize_gamma(gamma_range, reference_energy)
    
    def validate_with_experiment(self, experimental_energy: float, tolerance: float = 0.01) -> bool:
        """Сравнение с экспериментальным значением"""
        if 'VQE' not in self.results:
            self.calculate_energy('VQE')
        
        vqe_energy = self.results['VQE']['energy']
        error = abs(vqe_energy - experimental_energy)
        
        self.results['Validation'] = {
            'experimental': experimental_energy,
            'calculated': vqe_energy,
            'error': error,
            'within_tolerance': error <= tolerance
        }
        
        return error <= tolerance

# --------------------------
# Примеры использования
# --------------------------
if __name__ == "__main__":
    print("="*50)
    print(" Квантово-химический симулятор QuantumChemSimulator ")
    print("="*50)
    
    # Инициализация симулятора
    simulator = QuantumChemSimulator(basis_set='STO-3G', gamma=0.5)
    
    # Загрузка молекулы воды
    water = "O 0 0 0; H 0 0.76 -0.5; H 0 -0.76 -0.5"
    simulator.load_molecule(water)
    
    print("\nМолекула воды:")
    print(simulator.molecule)
    
    # Расчет энергии методом Хартри-Фока
    hf_energy = simulator.calculate_energy('HF')
    print(f"\nЭнергия Хартри-Фока: {hf_energy:.6f} Ha")
    
    # Расчет энергии методом VQE с анзацем UCCSD
    vqe_energy = simulator.calculate_energy('VQE', 'UCCSD')
    print(f"Энергия VQE (UCCSD): {vqe_energy:.6f} Ha")
    
    # Расчет гибридной энергии DFT+VQE
    hybrid_energy = simulator.calculate_energy('Hybrid')
    print(f"Гибридная энергия DFT+VQE: {hybrid_energy:.6f} Ha")
    
    # Исследование зависимости от базисного набора
    print("\nИсследование зависимости от базисного набора:")
    basis_sets = ['STO-3G', '6-31G']  # В реальности больше наборов
    basis_results = simulator.calculate_basis_dependence(basis_sets, 'VQE')
    for basis, energy in basis_results.items():
        print(f"{basis}: {energy:.6f} Ha")
    
    # Построение графика сходимости
    simulator.plot_convergence('VQE')
    
    # Оптимизация параметра gamma
    optimal_gamma, optimal_energy = simulator.optimize_gamma()
    print(f"\nОптимальный gamma: {optimal_gamma:.3f}, Энергия: {optimal_energy:.6f} Ha")
    
    # Валидация с экспериментальным значением (примерное значение для воды)
    is_valid = simulator.validate_with_experiment(-76.0)  # Примерное значение
    validation = simulator.results['Validation']
    print(f"\nВалидация с экспериментом:")
    print(f"Расчетное: {validation['calculated']:.6f} Ha, Эксперимент: {validation['experimental']} Ha")
    print(f"Ошибка: {validation['error']:.6f} Ha, В пределах допуска: {validation['within_tolerance']}")
    
    print("\nРасчет завершен успешно!")