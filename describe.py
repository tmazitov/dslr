import sys
import csv


def is_number(value):
    if value == '':
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_dataset(path):
    try:
        with open(path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
    except FileNotFoundError:
        sys.exit(f"describe error: file '{path}' not found")
    except StopIteration:
        sys.exit("describe error: dataset is empty")
    except OSError as e:
        sys.exit(f"describe error: {e}")
    return header, rows


def numeric_columns(header, rows):
    columns = {}
    for col_index, name in enumerate(header):
        if name == 'Index':
            continue
        values = []
        has_value = False
        all_numeric = True
        for row in rows:
            if col_index >= len(row):
                continue
            cell = row[col_index]
            if cell == '':
                continue
            has_value = True
            if is_number(cell):
                values.append(float(cell))
            else:
                all_numeric = False
                break
        if has_value and all_numeric:
            columns[name] = values
    return columns


def ft_count(values):
    count = 0
    for _ in values:
        count += 1
    return count


def ft_mean(values, count):
    total = 0.0
    for v in values:
        total += v
    return total / count


def ft_std(values, count, mean):
    if count <= 1:
        return 0.0
    total = 0.0
    for v in values:
        total += (v - mean) ** 2
    return (total / (count - 1)) ** 0.5


def ft_min(values):
    result = values[0]
    for v in values:
        if v < result:
            result = v
    return result


def ft_max(values):
    result = values[0]
    for v in values:
        if v > result:
            result = v
    return result


def ft_percentile(sorted_values, count, percent):
    if count == 1:
        return sorted_values[0]
    rank = percent * (count - 1)
    lower = int(rank)
    upper = lower + 1
    fraction = rank - lower
    if upper >= count:
        return sorted_values[lower]
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def compute_stats(columns):
    stats = {}
    for name, values in columns.items():
        count = ft_count(values)
        mean = ft_mean(values, count)
        std = ft_std(values, count, mean)
        sorted_values = sorted(values)
        stats[name] = {
            'Count': float(count),
            'Mean': mean,
            'Std': std,
            'Min': ft_min(values),
            '25%': ft_percentile(sorted_values, count, 0.25),
            '50%': ft_percentile(sorted_values, count, 0.50),
            '75%': ft_percentile(sorted_values, count, 0.75),
            'Max': ft_max(values),
        }
    return stats


def print_stats(stats):
    if not stats:
        sys.exit("describe error: no numerical features found in dataset")

    row_labels = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    feature_names = list(stats.keys())

    formatted = {
        name: {label: f"{stats[name][label]:.6f}" for label in row_labels}
        for name in feature_names
    }

    col_widths = {
        name: max(len(name), max(len(formatted[name][label]) for label in row_labels))
        for name in feature_names
    }

    label_width = max(len(label) for label in row_labels)

    header_line = ' ' * label_width
    for name in feature_names:
        header_line += '  ' + name.rjust(col_widths[name])
    print(header_line)

    for label in row_labels:
        line = label.ljust(label_width)
        for name in feature_names:
            line += '  ' + formatted[name][label].rjust(col_widths[name])
        print(line)


def main():
    argv = sys.argv
    if len(argv) != 2:
        sys.exit("describe error: invalid number of arguments")
    elif len(argv[1]) <= 4 or argv[1].find(".csv") == -1:
        sys.exit("describe error: invalid path to the dataset (.csv file)")

    header, rows = read_dataset(argv[1])
    columns = numeric_columns(header, rows)
    stats = compute_stats(columns)
    print_stats(stats)


if __name__ == '__main__':
    main()
