def get_decision(prob):
    if prob > 0.8:
        return "BLOCK 🚫"
    elif prob > 0.4:
        return "REVIEW ⚠️"
    else:
        return "ALLOW ✅"