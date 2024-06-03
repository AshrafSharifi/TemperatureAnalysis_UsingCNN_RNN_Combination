# # List of lists with given values
# values_CNN_LASTM = [
#     [4.712747573852539, 4.710043430328369, 0.8636213541030884, 1.6261687278747559],
#     [5.394148826599121, 5.391844272613525, 1.0305182933807373, 2.4491631984710693],
#     [5.389686584472656, 5.389348983764648, 1.0261943340301514, 2.4822564125061035],
#     [4.916418552398682, 4.916114807128906, 0.9380943179130554, 2.063993453979492],
#     [4.9600629806518555, 4.959767818450928, 0.9309988617897034, 1.9112985134124756]
# ]


# values_LSTM = [[5.200796127319336,5.200796127319336,0.9563296437263489,2.0445332527160645],
# [5.474417686462402,5.474417686462402,0.9994298815727234,2.1153810024261475],
# [4.800349235534668,4.800349235534668,0.8895102143287659,1.815619945526123],
# [5.053676128387451,5.053676128387451,0.943228542804718,2.0067451000213623],
# [4.8879594802856445,4.8879594802856445,0.9313923716545105,2.036611318588257],
# ]

root = 'data/Test/'
scripts = ['LSTM', 'CNN_FollowedBy_LSTM', 'LSTM_FollowedBy_CNN','CNN_Parrarel_LSTM']

for item in scripts:
    with open(root+'Test_'+item+'/outputResultFor5runs.txt', 'r') as file:
        values = file.read()
        
    # Split the string by newline characters
    lines = values.strip().split('\n')
    
    # Initialize an empty list to store the rows of floats
    float_rows = []
    
    # Process each line
    for line in lines:
        # Split the line by '__' and convert each part to float
        float_row = [float(num) for num in line.split('__') if num]
        # Append the list of floats to the rows list
        float_rows.append(float_row)
    
    # Initialize a list to store the sum of each column
    sums = [0] * 4
    
    for i in range(4):
        for sublist in float_rows:
            sums[i] += sublist[i]
        
    

    
    # Calculate the average of each column
    averages = [s / len(float_rows) for s in sums]
    
    factor = ['loss','mape', 'mae', 'mse']
    print('-----------------------------------------------------')
    print(item)
    # Print the averages
    for i, avg in enumerate(averages):
        print(factor[i] + f": {avg}")
    print('-----------------------------------------------------')
        
    
    
    
