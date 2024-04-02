def save_pareto_front(pareto_front, filename):
    with open(filename, 'w') as f:
        f.write(', '.join(["weight_{i}".format(i=i) for i in range(len(pareto_front[0][0]))]) + ", profit, risk\n")
        for individual in pareto_front:
            f.write(', '.join(str(weight) for weight in individual[0]) + ", {}, {}\n".format(individual[1], individual[2]))
