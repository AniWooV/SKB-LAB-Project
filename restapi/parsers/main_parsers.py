import camelot

class Analyze():
  def __init__(self, name, value, units):
        self.name = name
        self.value = value
        self.units = units


dataset = {
    'гемоглобин hgb': ['г/л', 'g/L', 'г/дл'],
    'эритроциты rbc': ['10^12/л', '10E12/L', 'млн/мкл'],
    'средний объем эритроцитов mcv': ['фемтолитр', 'fl', 'фл'],
    'среднее содержание гемоглобина в эритроците mchc': ['пкг', 'pg'],
    'тромбоциты plt': ['10^9/л', '10E9/L', 'тыс/мкл', 'x10^9/л'],
    'лейкоциты wbc': ['10^9/л', '10E9/L', 'тыс/мкл', 'x10^9/л'],
    'гематокрит hct': ['%'],
    'скорость оседания эритроцитов соэ':['мм/ч'],
    'нормобласты, %': ['%'],
    'базофилы, balso %': ['%'],
    'эозинофилы, абс. eos eo':['тыс/мкл ', '10^9/л'],
    'нейтрофилы, абс. neut ne':['%', '10^9/л', '10E9/L', 'тыс/мкл', 'x10^9/л'],
    'лимфоциты, % lymph lymf' : ['%'],
    'моноциты, % mon Mono':['%'],
}

def parse_pdf(directory):
    table = camelot.read_pdf(directory)

    table_list = table[0].df.values.tolist()

    analyzes =[]
    for row in table_list:
      for item in row:
          for key in  dataset.keys():
                if(item.lower() in key):
                  for i in row:
                    if(i.lower() in dataset[key]):
                      units = i
                      value = row[row.index(i)-1]
                      analyzes.append(Analyze(item.lower(), value, units))
                      break
                  break
          break  

    result = list()

    for i in analyzes:
      result.append([i.name, i.value, i.units])
            
    return result