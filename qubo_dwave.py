#!/usr/bin/env python3

from __future__ import print_function  # Kompatibilität mit Python 2 und 3
import numpy as np
from dwave.system import EmbeddingComposite, DWaveSampler
import dwave.inspector

# ============================
# 1. Erstellung der QUBO-Matrix
# ============================

def create_qubo_matrix():
    # Anzahl der Städte (z. B. A, B, C, D) und Tage
    n_cities = 4  # Städte: A, B, C, D
    n_days = 3    # Tage: Tag 1, Tag 2, Tag 3
    penalty_weight = 500  # Strafgewichtung für Constraints

    # Initialisiere die QUBO-Matrix (n_cities * n_days x n_cities * n_days)
    qubo_matrix = np.zeros((n_cities * n_days, n_cities * n_days))

    # Beispiel-Distanzen zwischen Städten (A=0, B=1, C=2, D=3)
    distances = {
        (0, 1): 100,  # Distanz zwischen A und B
        (0, 2): 200,  # Distanz zwischen A und C
        (0, 3): 300,  # Distanz zwischen A und D
        (1, 2): 150,  # usw.
        (1, 3): 250,
        (2, 3): 100
    }

    # Füge die linearen Terme (Kostenfunktion) hinzu
    for city in range(n_cities):
        for day in range(n_days):
            index = city * n_days + day
            qubo_matrix[index, index] = -distances.get((city, city), 0)

    # Füge die Strafterme für Constraints hinzu
    # Constraint 1: Jede Stadt wird genau einmal besucht
    for city in range(n_cities):
        for day1 in range(n_days):
            for day2 in range(day1 + 1, n_days):
                index1 = city * n_days + day1
                index2 = city * n_days + day2
                qubo_matrix[index1, index1] += penalty_weight
                qubo_matrix[index2, index2] += penalty_weight
                qubo_matrix[index1, index2] -= penalty_weight

    # Constraint 2: Jede Position (Tag) hat nur eine Stadt
    for day in range(n_days):
        for city1 in range(n_cities):
            for city2 in range(city1 + 1, n_cities):
                index1 = city1 * n_days + day
                index2 = city2 * n_days + day
                qubo_matrix[index1, index1] += penalty_weight
                qubo_matrix[index2, index2] += penalty_weight
                qubo_matrix[index1, index2] -= penalty_weight

    return qubo_matrix

# QUBO-Matrix erstellen und speichern
qubo_matrix = create_qubo_matrix()
np.savetxt('qubomatrix.txt', qubo_matrix, fmt='%.2f')
print("QUBO-Matrix wurde in 'qubomatrix.txt' gespeichert.")

# ============================
# 2. Laden der QUBO-Matrix aus Datei
# ============================
qubomatrix = np.loadtxt('qubomatrix.txt')
print('Geladene QUBO-Matrix:\n', qubomatrix, '\n')

# Konvertiere die Matrix in das QUBO-Dictionary-Format
qubo = {(i, i): qubomatrix[i, i] for i in range(len(qubomatrix))}
for i in range(len(qubomatrix)):
    for j in range(i + 1, len(qubomatrix)):
        if qubomatrix[i, j] != 0:
            qubo[(i, j)] = qubomatrix[i, j]

print('QUBO-Dictionary für D-Wave:\n', qubo, '\n')

# ============================
# 3. D-Wave Quantum Annealer nutzen
# ============================
# Verbinde mit dem D-Wave Sampler
sampler = EmbeddingComposite(DWaveSampler(token="DEV-44341c8fd341eb68c5bd3c35b314b31afad6d2b9"))

# Führe die Optimierung mit 1000 Reads durch
response = sampler.sample_qubo(qubo, num_reads=1000, chain_strength=2.0, annealing_time=20)
print('Antwort vom D-Wave:\n', response, '\n')

# Ergebnisse analysieren
best_solution = response.first.sample
energy = response.first.energy
print("Beste Lösung:", best_solution)
print("Gesamtkosten (Energie):", energy)

# ============================
# 4. Ergebnisse speichern
# ============================
# Ergebnisse in einer Datei speichern
with open('results.txt', 'w') as file:
    file.write('energy\tnum_occurrences\tsample\n')
    for sample, energy, num_occurrences, cbf in response.data():
        newsample = np.array([value for key, value in sorted(sample.items())])  # Neu sortieren
        file.write('%f\t%d\t%s\n' % (energy, num_occurrences, np.array2string(newsample).replace('\n', '')))
    print("Ergebnisse wurden in 'results.txt' gespeichert.")

# ============================
# 5. D-Wave Inspector starten (optional)
# ============================
dwave.inspector.show(response)
