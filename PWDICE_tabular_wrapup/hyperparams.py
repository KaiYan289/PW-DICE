def ini_hpp(name):
    hpp = {}
    f = open(name, "r")
    lines = f.readlines()
    for line in lines:
        contents = line.split()
        if contents[0] == "eval_use_argmax":
            hpp[contents[0]] = contents[1]
        elif contents[0] == "noise_level":
            hpp[contents[0]] = float(contents[1])
        else:
            hpp[contents[0]] = int(contents[1])
    return hpp
