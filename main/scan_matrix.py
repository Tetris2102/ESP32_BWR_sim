def scan_matrix(cols: list, rows: list, output_list: list):
    """
    Handle scanning of switch matrix of any size
    """

#     # Create output list of given size
#     output_list = [len(rows) * [False] for i in range(len(cols))]

    if not all(isinstance(row, list) for row in output_list):
        raise ValueError(f"output_list must be a list of lists (2D matrix): {output_list}")
    
    if any(isinstance(item, list) for item in lst):  # Not 1D list
        for col_idx, col in enumerate(cols):
            col.value(0)
            
            for row_idx, row in enumerate(rows):
                if row.value() == 0:
                    output_list[col_idx][row_idx] = True  # Switch on
                else:
                    output_list[col_idx][row_idx] = False  # Otherwise off
            
            col.value(1)  # Deactivate column
            time.sleep_us(500)  # Small delay between columns
    else:
        for col_idx, col in enumerate(cols):
            col.value(0)
            
            if rows.value() == 0:
                output_list[col_idx] = True
            else:
                output_list[col_idx] = False

def scan_switches(cols: list, rows: list):
    """
    Scan switch matrix and detect presses
    """
    global switch_states

    # Create list of lists to detect any rows with switches on
    switch_states_detect = switch_states
    scan_matrix(cols, rows, switch_states_detect)
    
    for i in range(len(cols)):
        for j in range(len(rows)):
            if switch_states_detect[i][j]:
                scan_matrix(cols, rows, switch_states[j])