import pprint

import numpy as np

# global variables that control debug/log formatting
DEBUG = True
VERBOSE = False
LOGFILE = "run.log"
LOGMODE = "file"        # "file" or "both"
TITLE_LENGTH = 76
DECIMALS = 8


# configure debug/log behavior from command-line flags
def set_debug(debug=False):
    global DEBUG, VERBOSE, LOGFILE, LOGMODE

    DEBUG = True
    VERBOSE = False
    LOGFILE = "run.log"
    LOGMODE = "both" if debug else "file"

    open(LOGFILE, "w", encoding="utf-8").close()


# send one log line to run.log and optionally to the terminal
def write_log_line(line):
    if LOGMODE in ("print", "both"):
        print(line)

    if LOGMODE in ("file", "both"):
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# convert logged objects into a consistent printable form
def format_logged_value(obj):
    if obj is None:
        return "None"

    if isinstance(obj, str):
        return obj

    if isinstance(obj, (float, np.floating)):
        return f"{obj:.{DECIMALS}f}"

    if isinstance(obj, (int, np.integer)):
        return str(obj)

    # print arrays/lists compactly
    try:
        arr = np.asarray(obj)
        if arr.ndim >= 1:
            return np.array2string(arr, precision=DECIMALS, suppress_small=True, threshold=200, edgeitems=3)
    except Exception:
        pass

    return pprint.pformat(obj, width=120)


# write a debug message with an optional value and unit label
def write_debug_message(msg, obj=None, units=None):
    if obj is None:
        write_log_line(msg)
        return

    formatted = format_logged_value(obj)

    if "\n" not in formatted:
        if units is None:
            write_log_line(f"  {msg} = {formatted}")
        else:
            write_log_line(f"  {msg} = {formatted} {units}")
    else:
        write_log_line(f"  {msg}:")
        for line in formatted.splitlines():
            write_log_line(f"    {line}")


# log a normal debug message
def log_debug_message(msg, obj=None, units=None):
    if DEBUG:
        write_debug_message(msg, obj=obj, units=units)


# log a verbose-only debug message
def log_verbose_message(msg, obj=None, units=None):
    if VERBOSE:
        write_debug_message(msg, obj=obj, units=units)


# print a blank log line for spacing
def log_blank_line():
    log_debug_message("")


# print a centered program header
def log_program_header(title):
    line = "=" * TITLE_LENGTH
    log_debug_message(line)
    log_debug_message(title.center(TITLE_LENGTH))
    log_debug_message(line)


# print a boxed section title
def log_section_header(title):
    inner_width = TITLE_LENGTH - 2
    if len(title) > inner_width:
        title = title[:inner_width]

    centered = title.center(inner_width)
    top = "+" + "-" * inner_width + "+"
    mid = "|" + centered + "|"

    log_debug_message("")
    log_debug_message(top)
    log_debug_message(mid)
    log_debug_message(top)


# log a step that the program is about to perform
def log_process_step(msg):
    log_debug_message(f"--> {msg}")


# return a default width for numeric table columns
def numeric_table_column_width(kind, precision=None):
    if precision is None:
        precision = DECIMALS
    if kind == "int":
        return 4
    if kind == "float":
        return 1 + 4 + 1 + precision
    return 0


# compute consistent column widths from headers and data types
def calculate_table_widths(labels, kinds=None, precision=None):
    if precision is None:
        precision = DECIMALS

    if kinds is None:
        kinds = ["str"] * len(labels)

    widths = []
    for label, kind in zip(labels, kinds):
        header_w = len(str(label))
        value_w = numeric_table_column_width(kind, precision=precision)
        widths.append(max(header_w, value_w))

    return widths


# format table cell to a fixed width
def format_table_cell(value, width, precision=None):
    if precision is None:
        precision = DECIMALS

    if value is None:
        return f"{'None':>{width}}"

    if isinstance(value, str):
        return f"{value:>{width}}"

    if isinstance(value, (int, np.integer)):
        return f"{value:>{width}d}"

    if isinstance(value, (float, np.floating)):
        return f"{value: {width}.{precision}f}"

    return f"{str(value):>{width}}"


# print header row and separator for a debug table
def write_table_header(labels, widths=None):
    if not DEBUG:
        return

    if widths is None:
        widths = calculate_table_widths(labels)

    header = " | ".join(f"{str(label):>{w}}" for label, w in zip(labels, widths))
    separator = "-+-".join("-" * w for w in widths)

    write_log_line(header)
    write_log_line(separator)


# print row of table values using fixed column width
def write_table_row(values, widths=None, precision=None):
    if not DEBUG:
        return

    if precision is None:
        precision = DECIMALS

    if widths is None:
        widths = [len(str(v)) for v in values]

    row = " | ".join(format_table_cell(val, w, precision=precision) for val, w in zip(values, widths))
    write_log_line(row)


# print structured debug summary of one atomic structure/state
def log_structure_summary(state, label="Structure"):
    log_section_header(label)

    log_debug_message("natoms", state.natoms())
    log_debug_message("lattice", state.lattice, units="Angstrom")
    log_debug_message("positions_cart", state.positions_cart, units="Angstrom")
    log_debug_message("positions_frac", state.positions_frac)

    if state.energy is not None:
        log_debug_message("energy", state.energy, units="eV")

    if state.forces is not None:
        force_norms = np.linalg.norm(state.forces, axis=1)
        log_debug_message("forces", state.forces, units="eV/Angstrom")
        log_debug_message("max_force", float(np.max(force_norms)), units="eV/Angstrom")

    if state.stress is not None:
        log_debug_message("stress", state.stress, units="eV/Angstrom^3")
    if state.pressure is not None:
        log_debug_message("pressure", state.pressure, units="eV/Angstrom^3")

    # metadata can be large, so only show it in verbose mode
    if state.metadata:
        log_verbose_message("metadata", state.metadata)
