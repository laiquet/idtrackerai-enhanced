def print_summary_matching(transfer_dicts, fragments_session, network_session):
    assignment_types = list(transfer_dicts.keys())
    header = "  |  ".join(assignment_types)

    print("{} -> {} : ".format(fragments_session, network_session) + header)

    ids = transfer_dicts["max_freq"]["assignments"].keys()
    for i in ids:
        assignments = (
            f"({transfer_dicts[assignment_type]['assignments'][i]},"
            f" {transfer_dicts[assignment_type]['assignments_values'][i]:.2f})"
            for assignment_type in assignment_types
        )
        print(f"{i} -> " + "  |  ".join(assignments))
