


def accuracy(predict:list[int], label:list[int]) -> float:
    match_num = sum([p == l for p,l in zip(predict, label)])
    print(f"accuracy:{match_num}/{len(predict)} : {match_num / len(predict)}")
    
    return match_num / len(predict)



def confusion_matrix():
    return 0.0