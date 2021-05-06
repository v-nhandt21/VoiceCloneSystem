import random
count =0
with open("MCDtest.txt","w+", encoding = "utf-8") as fw:
    fw.write("script\tref\tground\n")
    with open("prompts.txt","r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            path1, script1 = line.split(" ",1)
            cand = []
            for line2 in lines:
                #print(line)
                path2, script2 = line2.split(" ",1)

                if len(script1.split()) < 5 or len(script2.split()) < 5:
                    continue
                if path1 == path2:
                    continue

                spk1, _ = path1.split("_")
                spk2, _ = path2.split("_")

                if spk1 == spk2 and path1 != path2:
                    cand.append(path2)

            try:
                path2 = random.choice(cand)
                fw.write(script1+"\t"+path2+"\t"+path1+"\n")
                count = count +1
            except:
                print("No cand", path1 )

print(count)


