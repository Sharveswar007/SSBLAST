import cupyx.scipy.linalg as cl
lu_methods = [m for m in dir(cl) if "lu" in m.lower()]
print("cupyx lu methods:", lu_methods)
