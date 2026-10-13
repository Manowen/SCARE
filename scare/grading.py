import re
from math_verify import parse, verify, LatexExtractionConfig

def final_box(text):
    p = text.rfind('\\boxed{')
    if p < 0:
        m = re.findall('####\\s*([-+−]?[\\d][\\d,. ]*)', text)
        return m[-1].strip() if m else None
    start = p + 7
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return text[start:i]
    return None

def grade(text, gold):
    value = final_box(text)
    if value is None:
        return (False, None)
    try:
        g = parse('$' + str(gold).strip().strip('$') + '$', extraction_config=[LatexExtractionConfig()], fallback_mode='no_fallback')
        p = parse('$' + value.strip().strip('$') + '$', extraction_config=[LatexExtractionConfig()], fallback_mode='no_fallback')
        return (bool(g and p and verify(g, p)), value if p else None)
    except Exception:
        return (False, None)
