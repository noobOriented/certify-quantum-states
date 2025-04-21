import cmath
import random
import sys
import typing as t

n: int  # system size

# Target state is the output of a 1D IQP circuit:
# HHH...HHH (\Pi_i T^{random_T_pattern[i]}) (\Pi_i CZ_{i, i+1}) HHH...HHH |000...000>
# 
# We consider random_T_pattern[i] to be {-1, 0, 1}

random_T_pattern: list[int] = []

# Integer for each action:
# 	 0 * n, ..., 1 * n - 1: HH CZ HH
# 	 1 * n, ..., 2 * n - 1: H T H
# 	 2 * n, ..., 3 * n - 1: H T^-1 H

def main():
	if len(sys.argv) < 4:
		print('Usage: ./StatePrep <n> <seed> <type>')
		return

	# Type = 0: Train with fidelity
	# Type = 1: Train with shadow overlap
	# Type = 2: Plot state construction process
	global n
	n, seed, typ = map(int, sys.argv[1:])
	random.seed(seed)

	for _ in range(n):
		random_T_pattern.append(random.randint(-1, 1))

	seq_action: list[int] = []

	if typ < 2:
		cur_score = -9999
		cur_fidel = 0.0
		cur_shadow_o = 0.0

		for _ in range(3 * n):
			best_act = 0
			best_one_step = -9999
			best_one_step_fidel = 0.0
			best_one_step_shadow_o = 0.0

			for act in range(3 * n):
				seq_action.append(act)

				shadow_o = sum(
					estimate_one_shadow_overlap(seq_action) / 10000
					for _ in range(10000)
				)

				fidel = estimate_fidelity(seq_action)
				score = fidel if typ == 0 else shadow_o

				if best_one_step < score:
					best_one_step = score
					best_act = act
					best_one_step_fidel = fidel
					best_one_step_shadow_o = shadow_o

				seq_action.pop()

			if cur_score < best_one_step or cur_score < 0.99:
				seq_action.append(best_act)
				cur_score = best_one_step
				cur_fidel = best_one_step_fidel
				cur_shadow_o = best_one_step_shadow_o

			print(f'{cur_fidel} {2 * (cur_shadow_o - 0.5)}')

	else:
		for i in range(n):
			seq_action.append(i)
			if random_T_pattern[i] != 0:
				seq_action.append(((random_T_pattern[i] + 3) % 3) * n + i)

			shadow_o = sum(  # TODO mean
				estimate_one_shadow_overlap(seq_action) / 10_000
				for _ in range(10_000)
			)
			fidel = estimate_fidelity(seq_action)
			print(f'{fidel} {2 * (shadow_o - 0.5)}')

	return 0


def estimate_fidelity(seq_action: t.Sequence[int]) -> float:
	fidel = 0.0
	for _ in range(10_000):
		bitstring = random.choices([0, 1], k=2)

		how_many_T = 0
		for act in seq_action:
			T_or_not, pos = divmod(act, n)

			if T_or_not == 1:
				if bitstring[pos] == 1:
					how_many_T += 1
			elif T_or_not == 2:
				if bitstring[pos] == 1:
					how_many_T = (how_many_T - 1 + 8) % 8

			else:
				if bitstring[pos] == 1 and bitstring[(pos + 1) % n] == 1:
					how_many_T += 4

		how_many_T_true = 0
		for i in range(n):
			if bitstring[i] == 1 and bitstring[(i + 1) % n] == 1:
				how_many_T_true += 4
			if bitstring[i] == 1:
				how_many_T_true = (how_many_T_true + random_T_pattern[i] + 8) % 8

		phase_diff = (how_many_T - how_many_T_true + 8) % 8
		fidel += cmath.exp(1j * (phase_diff / 8.0) * 2.0 * cmath.pi) / 10000.0

	return abs(fidel) * abs(fidel)


def estimate_one_shadow_overlap(seq_action: t.Sequence[int]) -> float:
	"""Shadow overlap is measured in all X bases except for qubit random_x"""
	random_x = random.randrange(n)
	bitstring = [
		random.randrange(2) if i != random_x else 0
		for i in range(n)
	]

	how_many_T = 0
	for act in seq_action:
		T_or_not, pos = divmod(act, n)
		if T_or_not == 1:
			if pos == random_x:
				how_many_T += 1

		elif T_or_not == 2:
			if pos == random_x:
				how_many_T = (how_many_T - 1 + 8) % 8
		else:
			if pos == (random_x - 1 + n) % n and bitstring[(random_x - 1 + n) % n] == 1:
				how_many_T += 4
			if pos == random_x and bitstring[(random_x + 1) % n] == 1:
				how_many_T += 4

	#
	# A simplified 1D cycle MPS contraction
	#
	# The contracted target 1-qubit state on the random_X qubit is
	#    1/sqrt(2) |0> + 1/sqrt(2) math.exp(i 2 pi (how_many_T_true/8)) |1>
	#
	# This is obtained by the structure of the target state
	#
	how_many_T_true = 0
	if bitstring[(random_x - 1 + n) % n] == 1:
		how_many_T_true += 4
	if bitstring[(random_x + 1) % n] == 1:
		how_many_T_true += 4

	how_many_T_true = (how_many_T_true + random_T_pattern[random_x] + 8) % 8
	phase_diff = (how_many_T - how_many_T_true + 8) % 8
	return (
		abs(0.5 + 0.5 * cmath.exp(1j * (phase_diff / 8.0) * 2.0 * cmath.pi))
		* abs(0.5 + 0.5 * cmath.exp(1j * (phase_diff / 8.0) * 2.0 * cmath.pi))
	)


if __name__ == '__main__':
	main()
