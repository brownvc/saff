def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    

    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # loading
    assert False, "load method"

    assert False, "load data"

    assert False, "load task"

    # evaluate
    assert False, "use loaded method to generate corresponding result for data and task on disk"
    assert False, "compare result with gt using metrics and save on disk"

    # save results
    assert False, "visualize and save quantitative results"
    assert False, "visualize and save qualitative results"
