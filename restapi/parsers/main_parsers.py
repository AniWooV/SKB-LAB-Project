import camelot


def get_parsed_analysis_with_camelot(file_name):
    table = camelot.read_pdf(file_name)

    table_list = table[0].df.values.tolist()

    return table_list
