lower_modifiers = [0x941,0x942, 0x943,0x094D] 
upper_modifiers = [0x901, 0x902]
core_modifiers = [0x915, 0x916, 0x917, 0x918, 0x919, 0x91a, 0x91b, 0x91c, 0x91d, 0x91e, 0x91f, 0x920, 0x921, 0x922, 0x923, 0x924, 0x925, 0x926, 0x927, 0x928, 0x92a, 0x92b, 0x92c, 0x92d, 0x92e, 0x92f, 0x930, 0x932, 0x935, 0x936, 0x937, 0x938, 0x939,0x905,0x907,0x909,0x90a,0x90b,0x90f]

def generate_word(unicode_list):
    return ''.join([chr(code) for code in unicode_list])
def stringInterpolate(coreMod,upperMod=10):
    if coreMod == 68:
        if upperMod == 0:
            return [0x940]
        elif upperMod == 1:
            return [0x94B]
        elif upperMod == 2:
            return [0x94C]
        elif upperMod == 10:
            return [0x93e]
    elif upperMod != 10:
        res = []
        if upperMod == 1:
            res = [0x947]
        elif upperMod == 2:
            res = [0x948]
        elif upperMod == 3:
            res = [0x901]
        elif upperMod == 4:
            res =[0x902]
        
        if coreMod == 33:
            return [0x915, 0x094D, 0x937] + res
        elif coreMod == 34:
            return [0x924, 0x094D, 0x930]+ res
        elif coreMod == 35:
            return [0x91c, 0x094D, 0x92f]+ res
        elif coreMod < 33:
            return [core_modifiers[coreMod]]+ res
        elif 35 < coreMod < 42:
            return [core_modifiers[coreMod]-3]+ res
    elif upperMod == 10:
        if coreMod == 33:
            return [0x915, 0x094D, 0x937]
        elif coreMod == 34:
            return [0x924, 0x094D, 0x930]
        elif coreMod == 35:
            return [0x91c, 0x094D, 0x92f]
        elif coreMod < 33:
            return [core_modifiers[coreMod]]
        elif 35 < coreMod < 42:
            return [core_modifiers[coreMod]-3]
        elif 41 < coreMod < 68:
            return [core_modifiers[coreMod]-41,0x094D]
    
