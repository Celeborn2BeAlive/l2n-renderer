if __name__ == "__main__":
    max_elt_count = 65536
    elt_count = 0
    with open("tinymt32dc.0.1048576.cpp", "w") as of:
        of.write("static tinymt32_params precomputed_tinymt_params[] = {\n")
        with open("tinymt32dc.0.1048576.txt", "r") as f:
            for l in f.readlines():
                if l.startswith("#"):
                    continue
                p = l.split(",")
                of.write("    { 0x" + p[3] + ", 0x" + p[4] + ", 0x" + p[5] + " }, \n")
                elt_count = elt_count + 1
                if elt_count >= max_elt_count:
                    break
        of.write("};\n")
            