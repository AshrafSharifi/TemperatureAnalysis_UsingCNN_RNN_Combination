sensor = 1

if sensor == 0:
    root = 'data/Test/'
else:
    root = 'data/Test/ForOneSensor/'
    
    
    

scripts = ['LSTM', 'CNN_FollowedBy_LSTM', 'LSTM_FollowedBy_CNN','CNN_Parrarel_LSTM']

for item in scripts:
    if sensor == 0:
        filepath = root+'Test_'+item+'/outputResultFor5runs.txt'
    else:
        filepath = root+'Test_'+item+'/Sensor'+str(sensor)+'/outputResultFor5runs.txt'
        
    with open(filepath, 'r') as file:
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
        
    
    
    
