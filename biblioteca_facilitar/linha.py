def linha(padrao="-=", vezes=20):
    for i in range(vezes):
        if i != vezes - 1:
            print(padrao, end="")
        else:
            print(padrao)