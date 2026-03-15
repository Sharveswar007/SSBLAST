from triton.backends import backends
for k, b in backends.items():
    print(k, b.driver.is_active())
