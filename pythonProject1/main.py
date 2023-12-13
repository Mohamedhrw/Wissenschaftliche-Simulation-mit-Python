import numpy as np
import pandas as pd
import math
#  prüfen, ob Matrix quadratisch ist
def is_square(matrix):
    rows, cols = matrix.shape
    return rows == cols

# prüfen, ob Matrix symmetrisch ist
def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

def validate_input(matrix):
    # Überprüfen der Matrix
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("Matrix enthält ungültige Elemente. Es dürfen nur Zahlen verwendet werden.")

# Funktion zur Durchführung der Cholesky-Zerlegung
def cholesky_decomposition(A):
    if not is_square(A):
        raise ValueError(" Matrix erfüllt nicht die Voraussetzungen ( nicht quadratisch). ")
    if not is_symmetric(A):
        raise ValueError(" Matrix erfüllt nicht die Voraussetzungen ( nicht symmetrisch).")

    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                temp = A[i, i] - np.sum(L[i, :i] ** 2)
                if temp <= 0:
                    raise ValueError("Matrix erfüllt nicht die Voraussetzungen (positiven Definitheit).")
                L[i, i] = np.sqrt(temp)
            else:
                temp = A[i, j] - np.dot(L[i, :j], L[j, :j])
                L[i, j] = temp / L[j, j]

    return L

# Aufgabe (ii): Lineares Ausgleichsproblem lösen

# berechnet die Lösung für das kleinste-Quadrate-Problem mit Hilfe der Cholesky-Zerlegung.
def calculate_least_squares_solution(matrix_A, vector_b):

    product_A = np.dot(matrix_A.T, matrix_A) # das Produkt A^T * A berechnen
    L = cholesky_decomposition(product_A)
    solution_y = np.linalg.solve(L, np.dot(matrix_A.T, vector_b)) # das Gleichungssystem L * y = A^T * b lösen
    solution_x = np.linalg.solve(L.T, solution_y) # das Gleichungssystem L^T * x = y lösen

    return solution_x

# Aufgabe 3.1

# Diese Funktion führt die Householder-Transformation auf einer gegebenen Matrix durch.
# Die Householder-Transformation verwendet Householder-Vektoren,
# um bestimmte Spaltenvektoren in der Matrix zu nullen und dabei die Matrizen R, H und Q zu aktualisieren.
# Die Funktion gibt  die transponierte Matrix Q und die obere Dreiecksmatrix R zurück.
def householder_transformation(A):
	# Die Dimensionen ermitteln
	m, n = A.shape

	R = np.copy(A).astype('float64')  # Datentyp als float64 festlegen
	Q = np.eye(m)
	H = np.eye(m)
	for k in range(n):
		# Householder-Vektor u erstellen
		u = np.copy(R[k:, k])

		# Update des ersten Eintrags von u
		u[0] = u[0] + np.sign(u[0]) * np.linalg.norm(u)

		# Normalisierung des Householder-Vektors u
		u = u / np.linalg.norm(u)
		R[k:, k:] -= 2 * np.outer(u, np.dot(u, R[k:, k:]))
		H[k:, k:] -= 2 * np.outer(u, np.dot(u, H[k:, k:]))
		Q[k:] -= 2 * np.outer(u, np.dot(u, Q[k:]))
	return H, Q.T, R

# Diese Funktion führt Givens-Rotationen auf einer gegebenen Matrix durch.
# Givens-Rotationen werden verwendet, um bestimmte Elemente in der Matrix auf Null zu setzen und dabei die Matrix R,
# sowie die Matrizen H und Q zu aktualisieren, die die kumulativen Transformationen enthalten.
# Die Funktion gibt die transformierte Matrix H, die transponierte Matrix Q und die obere Dreiecksmatrix R zurück.

def givens_rotation(matrix_A):
    anzahl_zeilen, anzahl_spalten = matrix_A.shape
    Q = np.eye(anzahl_zeilen)  # Einheitsmatrix mit gleicher Größe wie A
    R = matrix_A.copy()  # Kopie der Matrix A, um sie nicht direkt zu ändern

    for j in range(anzahl_spalten):
        # Diese Schleife arbeitet Spalte für Spalte von links nach rechts und führt
        # für jede Spalte eine Rotation auf die entsprechenden Elemente durch.

        for i in range(anzahl_zeilen - 1, j, -1):
            a = R[i - 1, j]  # Element a in der Givens-Rotation
            b = R[i, j]  # Element b in der Givens-Rotation
            c = np.sqrt(a**2 + b**2)  # Berechnung der Hypotenuse c

            if c != 0:
                s = b / c  # Berechnung des Sinus s
                c = a / c  # Berechnung des Kosinus c
            else:
                s = 0  # Falls c = 0, setzen von s auf 0
                c = 1  # Falls c = 0, setzen von c auf 1

            # Berechnung der Givens-Rotation G
            G = np.array([[c, s], [-s, c]])

            # Anwendung von G auf R
            temp = np.zeros_like(R[i - 1:i + 1, j:anzahl_spalten])
            temp[0, :] = c * R[i - 1, j:anzahl_spalten] + s * R[i, j:anzahl_spalten]
            temp[1, :] = -s * R[i - 1, j:anzahl_spalten] + c * R[i, j:anzahl_spalten]
            R[i - 1:i + 1, j:anzahl_spalten] = temp

            # Anwendung von G auf Q
            temp = np.zeros_like(Q[i - 1:i + 1, :])
            temp[0, :] = c * Q[i - 1, :] + s * Q[i, :]
            temp[1, :] = -s * Q[i - 1, :] + c * Q[i, :]
            Q[i - 1:i + 1, :] = temp

    return Q.T, Q, R


# Diese Funktion löst das reduzierte Gleichungssystem Rx = b1 durch Rückwärtsiteration.
# Die Funktion erwartet eine obere Dreiecksmatrix R und den Vektor b1.
# Die Rückwärtsiteration wird verwendet, um die Lösung x zu berechnen, indem die Gleichungen R[i, i]*x[i] = b1[i] für i von n-1 bis 0 gelöst werden.
# Die Lösung wird als Vektor x zurückgegeben.
def solve_reduced_system(R, b1):
    n = R.shape[1]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if R[i, i] != 0:
            x[i] = (b1[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
        else:
            x[i] = 0
    return x  # Außerhalb der Schleife platzieren


	    # Aufgabe 4

	    # Beispielmatrizen und -vektoren
examples = [
        {
			'A': np.array([[3, -9, 7], [-4, -13, -1], [0, -20, -35]]),
			'b': np.array([1, 3, 2])
		},
        {
			'A': np.array([[1, 2, 1], [2, 4, 3], [1, 3, 6]]),
			'b': np.array([1, 2, 1])
		},
		{
			'A': np.array([[1, 2, 1], [2, 5, 2], [1, 2, 10]]),
			'b': np.array([0, 2, 1])
		},
		{
			'A': np.array([[2, 1, 1], [1, 2, 1], [1, 1, 5]]),
			'b': np.array([5, 5, 5])
		},
		{
			'A': np.array([[1, 2, 3], [-1, 2, -1], [0, -1, 2]]),
			'b': np.array([3, 1, 4])
		},
		{
			'A': np.array([[4, 2], [2, 5]]),
			'b': np.array([1, 2])
		},
]
# eine Matrix um residum Werte speicher und später tabellarisch darstellen
residum_results = []

	# Testen der Cholesky-Zerlegung und Lösung des Ausgleichsproblems
for i, example in enumerate(examples):
		print("-------------------------------------------------------------------")
		print(f"Beispiel {i + 1}:")
		print("-------------------------------------------------------------------")
		A = example['A']
		b = example['b']

		print("Matrix A:")
		print(A)
		print("Vektor b:")
		print(b)

		try:
			# Cholesky-Zerlegung
			L = cholesky_decomposition(A)
			print("Cholesky-Zerlegung:")
			print("L =")
			print(L)
			print("A = L * L^T:")
			print(np.dot(L, L.T))

			# Ausgleichsproblem lösen
			solution_x = calculate_least_squares_solution(A, b)
			print("Lösung des Ausgleichsproblems:")
			print("x =")
			print(format(solution_x.round(2)))
		except ValueError as e:
			print("Es ist ein Fehler aufgetreten:", str(e))

		print()

	# Testen der Householder-Transformation und Givens-Rotation
for i, example in enumerate(examples):
		print("-------------------------------------------------------------------")
		print(f"Beispiel {i + 1}:")
		print("-------------------------------------------------------------------")
		result = {}
		result['Beispiel'] = i + 1
		A = example['A']
		b = example['b']

		print("Matrix A:")
		print(A)
		print("Vektor b:")
		print(b)

		try:
			# Householder-Transformation
			H_householder, Q_householder, R_householder = householder_transformation(A)
			print("Householder-Transformation:")
			print("H:")
			print(H_householder.round(2))
			print("Q:")
			print(Q_householder.round(2))
			print("R:")
			print(R_householder.round(2))

			# Givens-Rotation
			QT_givens, Q_givens, R_givens = givens_rotation(A)
			print("\nGivens-Rotation:")
			print("QT:")
			print(QT_givens.round(2))
			print("Q:")
			print(Q_givens.round(2))
			print("R:")
			print(R_givens.round(2))

			# Reduziertes Gleichungssystem lösen
			b1_householder = np.dot(Q_householder.T, b)[:R_householder.shape[1]]
			x_householder = solve_reduced_system(R_householder, b1_householder)
			print("\nLösung des reduzierten Gleichungssystems (Householder):")
			print("x:")
			print(x_householder.round(2))

			b1_givens = np.dot(Q_givens.T, b)[:R_givens.shape[1]]
			x_givens = solve_reduced_system(R_givens, b1_givens)
			print("\nLösung des reduzierten Gleichungssystems (Givens):")
			print("x:")
			print(x_givens.round(2))
			# Residuum berechnen
			residual_householder = np.linalg.norm(np.dot(R_householder, x_householder) - b1_householder)
			result['Residuum (Householder)'] = residual_householder
			residual_givens = np.linalg.norm(np.dot(R_givens, x_givens) - b1_givens)
			result['Residuum (Givens)'] = residual_givens
			print("\nResiduum:")
			print("Householder:", residual_householder.round(16))
			print("Givens:", residual_givens.round(16))

		except ValueError as e:
			print("Es ist ein Fehler aufgetreten:", str(e))
		residum_results.append(result)
# Tabelle der Ergebnisse erstellen
df = pd.DataFrame(residum_results)
df = df.set_index('Beispiel')

# Tabelle mit Residuum-Werten anzeigen
residuum_df = df.filter(regex=r'Residuum')
print(residuum_df)

print()
