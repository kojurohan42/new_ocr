# Devanagari Unicode range: U+0900 to U+097F
start_unicode = 0x0900
end_unicode = 0x097F

devanagari_characters = [(chr(code), hex(code)) for code in range(start_unicode, end_unicode + 1)]

for char, code in devanagari_characters:
    print(f"Character: {char}, Unicode: {code}")

for char, code in devanagari_characters:
    print(code)
    # print(f"Character: {char}, Unicode: {code}")
lower_modifiers = []

def generate_word(code_points):
    word = ""
    for code_point in code_points:
        word += chr(code_point)
    return word


code_points = [0x0926, 0x0935, 0x093F, 0x091C,0x094D]  # Unicode code points for 'द', '्', 'वि', 'ज'
print(generate_word(code_points))

# def stringInterpolate(lowerMod, upperMod , coreMod):
#     if(coreMod == 68):
#         switch(upperMod):
    