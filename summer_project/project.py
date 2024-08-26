# Python program to download historical data CSV files of all 50 stocks of Nifty50 constituents of any timeframe
# from NSE website and find Standard Deviation and mean of each stock and return list
# Create and Solve LPP, Optimization Model include Optimized Weights, Optimized Portfolio Standard Deviation
# Amount to Invest in various stocks and plot a pie chart
# A: tayalankit20@gamil.com
# Download the following file, Open and correct it (remove newline \n in column name)
# https://www.nseindia.com/api/equity-stockIndices?csv=true&index=NIFTY%2050
# Recommended to view and edit in VScode (use ctrl+Alt for editing nifty50 file)

# importing the package numpy as no
import numpy as no
# importing the package pandas in shortform pd
import pandas as pd
# Import libraries
from matplotlib import pyplot as plt
# from scipy.optimize import linprog
from scipy.optimize import minimize

def nse_download():
    # importing the package for CSV download
    import webbrowser
    import time
    for i in symbol:
        nseurl = f'https://www.nseindia.com/api/historical/cm/equity?symbol={i}&series=[%22EQ%22]&from=24-09-2022&to=24-09-2023&csv=true'
        print(nseurl)
        # then call the default open method described above
        webbrowser.open(nseurl, new=2, autoraise=True)
        time.sleep(0.5)
        # with open("output.txt", "a") as f:
        #     print(nseurl, file=f)

    # Error M&M url dosent open correctly in loop
    webbrowser.open(
        'https://www.nseindia.com/api/historical/cm/equity?symbol=M%26M&series=[%22EQ%22]&from=24-09-2022&to=24-09-2023&csv=true', new=2, autoraise=True)
    print("Waiting to finish all downloading")
    time.sleep(5)


# FOR LOOP for all 50 Stocks
csv0 = pd.read_csv('MW-NIFTY-50-24-Sep-2023.csv')
df0 = pd.DataFrame(csv0, columns=['SYMBOL '])
# Convert DataFrame column as a list
symbol = (df0['SYMBOL '].tolist())
symbol.pop(0)
# print(symbol)
n = 0
sd = []
mean = []

# Un-comment this function to download CSV of all 50 stocks from NSE website
# nse_download()
# if ERROR: move the Nifty 50 file named 'MW-NIFTY-50-24-Sep-2023.csv' to same folder as this file
# Also correct the column name new line

for i in symbol:
    # reading the CSV file
    try:
        csv1 = pd.read_csv(f'Quote-Equity-{i}-EQ-24-09-2022-to-24-09-2023.csv')
    except FileNotFoundError as e:
        print(
            f"FileNotFoundError: {i} CSV file is missing. ***********---------********")
        continue
    # displaying the contents of the CSV file
    # print(csv1)
    # print(f'sucess open {i}')
    df = pd.DataFrame(csv1, columns=['OPEN ', 'close '])
    # print("the output datype is: of stock-", i)
    # print(df.dtypes)
    # print(df.dtypes['OPEN '])
    try:
        if df.dtypes['OPEN '] == 'object' or df.dtypes['close '] == 'object':
            # df['OPEN '] = dfOPEN str.replace(',', '').astype(int)
            # print("converting")
            # remove , comma from values
            df['OPEN '] = df['OPEN '].str.replace(',', '').astype(float)
            df['close '] = df['close '].str.replace(',', '').astype(float)
    except AttributeError as e:
        print(
            f'AttributeError: SUCESSFULLY HANDELED - Stock {i} has mixed datatypes in same column')
        print(df.dtypes)
    # return per day close - open / close
    df['result'] = ((df['close ']-df['OPEN '])/df['OPEN '])*100
    # Un-comment to show entire working dataset i.e OPEN, close, result
    # print(df)
    # Un-comment to show Retun per day column
    # print(df['result'])
    # Un-comment to show mean
    print(f'S.No {n+1} - The Mean of {i} is ', end="")
    print(df['result'].mean())
    mean.append(df['result'].mean())
    # Creating an array by making use of array function in NumPy and storing it in a variable called arrayname
    arrayname = no.array(df['result'])
    # Displaying the elements of arrayname followed by one line space by making use of \n
    # print('The elements of the given array are:')
    # print(arrayname)
    # using std function of NumPy and passing the created array as the parameter to that function to find the standard deviation value of all the elements in the array and store it in a variable called stddev
    stddev = no.std(arrayname)
    # Displaying the standard deviation value stored in stddev variable
    # print('The standard deviation of all the elements of the array is:')
    print(f'S.No {n+1} - The S.D of {i} is {stddev}')
    sd.append(stddev)
    n += 1

# print(f'\n \n The standard deviation of all {n} stocks are {sd} \n')
# print(f'\n \n The mean of all {n} stocks are {mean} \n')
mean.sort()
print(mean)
# print("LPP equaton")
# n = 1
# while (n < 50):
#     print("\n\nMaximize the following Linear Programming Problem \n\nMaximize: ")
#     print("Z = ", end="")
#     for i in mean:
#         print(f'w{n} * {i} + ', end="")
#         n +=1
#     print("\b\b .\n\nSubject to: \n")
#     n = 1
#     for i in mean:
#         print(f'w{n} + ', end="")
#         n +=1
#     print("\b\b= 1 ")
#     n = 1
#     for i in mean:
#         print(f'w{n} ≥ 0, ', end="")
#         n +=1
#     print('\b\b .')


# print("\n\n solving above eq\n")
# # Coefficients of the objective function
# c = [ -x for x in mean]
# # Coefficients of the equality constraint (sum of weights should be 1)
# A_eq = [[1] * 50]  # 1 for each of the 50 variables
# # RHS value of the equality constraint
# b_eq = [1]
# # Bounds for each variable (w1, w2, ..., w50)
# bounds = [(0, None) for _ in range(50)]  # Lower bound of 0 for all variables
# # Solve the linear programming problem
# result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
# # Extract the optimal solution
# optimal_weights = result.x
# # Extract the maximum value of the objective function (negate it since linprog minimizes by default)
# max_value = -result.fun
# print("Optimal Weights:", optimal_weights)
# print("Maximum Value (Z):", max_value)


def meancal(i):
    try:
        csv1 = pd.read_csv(f'Quote-Equity-{i}-EQ-24-09-2022-to-24-09-2023.csv')
    except FileNotFoundError as e:
        print(
            f"FileNotFoundError: {i} CSV file is missing. ***********---------********")
    df = pd.DataFrame(csv1, columns=['OPEN ', 'close '])
    try:
        if df.dtypes['OPEN '] == 'object' or df.dtypes['close '] == 'object':
            df['OPEN '] = df['OPEN '].str.replace(',', '').astype(float)
            df['close '] = df['close '].str.replace(',', '').astype(float)
    except AttributeError as e:
        print(
            f'AttributeError: SUCESSFULLY HANDELED - Stock {i} has mixed datatypes in same column')
        print(df.dtypes)
    df['result'] = ((df['close ']-df['OPEN '])/df['OPEN '])*100
    print(f'The Mean of {i} is ', end="")
    meanc = df['result'].mean()
    print(meanc)
    # print(df)
    return meanc


def covariance(i, j):
    try:
        csv1 = pd.read_csv(f'Quote-Equity-{i}-EQ-24-09-2022-to-24-09-2023.csv')
    except FileNotFoundError as e:
        print(
            f"FileNotFoundError: {i} CSV file is missing. ***********---------********")
    df = pd.DataFrame(csv1, columns=['OPEN ', 'close '])
    try:
        if df.dtypes['OPEN '] == 'object' or df.dtypes['close '] == 'object':
            df['OPEN '] = df['OPEN '].str.replace(',', '').astype(float)
            df['close '] = df['close '].str.replace(',', '').astype(float)
    except AttributeError as e:
        print(
            f'AttributeError: SUCESSFULLY HANDELED - Stock {i} has mixed datatypes in same column')
        print(df.dtypes)
    df['result'] = ((df['close ']-df['OPEN '])/df['OPEN '])*100
    # print(f'The Mean of {i} is ', end="")
    meanc = df['result'].mean()
    # print(meanc)
    df['rorm'] = df['result'] - meanc
    # print(df)
    try:
        csv2 = pd.read_csv(f'Quote-Equity-{j}-EQ-24-09-2022-to-24-09-2023.csv')
    except FileNotFoundError as e:
        print(
            f"FileNotFoundError: {j} CSV file is missing. ***********---------********")
    df1 = pd.DataFrame(csv2, columns=['OPEN ', 'close '])
    try:
        if df1.dtypes['OPEN '] == 'object' or df1.dtypes['close '] == 'object':
            df1['OPEN '] = df1['OPEN '].str.replace(',', '').astype(float)
            df1['close '] = df1['close '].str.replace(',', '').astype(float)
    except AttributeError as e:
        print(
            f'AttributeError: SUCESSFULLY HANDELED - Stock {j} has mixed datatypes in same column')
        print(df1.dtypes)
    df1['result'] = ((df1['close ']-df1['OPEN '])/df1['OPEN '])*100
    # print(f'The Mean of {j} is ', end="")
    meancc = df1['result'].mean()
    # print(meancc)
    df1['rorm'] = df1['result'] - meancc
    df1['mul'] = df['rorm'] * df1['rorm']
    covvf = df1['mul'].mean()
    # print(df1)
    # print(f'The covariance of {i},{j} is {covvf}')
    return covvf


def model(n1, n2, n3):
    # Mean returns
    mean_return_R = meancal(n1)
    mean_return_T = meancal(n2)
    mean_return_L = meancal(n3)

    # Covariance matrix
    # Calculating covariance between two variables X and Y: Cov(X, Y) = Σ[(X_i - Mean(X)) * (Y_i - Mean(Y))] / (Number of observations - 1)
    cov_matrix = [
        [covariance(n1, n1), covariance(n1, n2), covariance(n1, n3)],
        [covariance(n1, n2), covariance(n2, n2), covariance(n2, n3)],
        [covariance(n1, n3), covariance(n2, n3), covariance(n3, n3)]
    ]
    print('The Covariance matrix is \n', no.matrix(cov_matrix))

    # Negative of the objective function to minimize
    def negative_sharpe_ratio(weights):
        portfolio_return = sum([mean_return_R * weights[0], mean_return_T * weights[1], mean_return_L * weights[2]])
        portfolio_stddev = (weights[0] ** 2 * cov_matrix[0][0] + weights[1] ** 2 * cov_matrix[1][1] + weights[2] ** 2 * cov_matrix[2][2] + 2 * weights[0] * weights[1] * cov_matrix[0][1] + 2 * weights[0] * weights[2] * cov_matrix[0][2] + 2 * weights[1] * weights[2] * cov_matrix[1][2]) ** 0.5
        sharpe_ratio = portfolio_return / portfolio_stddev
        return -sharpe_ratio

    # Equality constraint (sum of weights == 1)
    def constraint(weights):
        return sum(weights) - 1

    # Initial guess for weights
    initial_weights = [0.33, 0.33, 0.34]  # Equal initial allocation

    # Bounds for weights
    bounds = ((0, 1), (0, 1), (0, 1))  # Each weight between 0 and 1

    # Equality constraint
    constraints = {'type': 'eq', 'fun': constraint}

    # Solve the optimization problem
    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract optimized weights
    optimized_weights = result.x
    optimized_portfolio_stddev = (optimized_weights[0] ** 2 * cov_matrix[0][0] + optimized_weights[1] ** 2 * cov_matrix[1][1] + optimized_weights[2] ** 2 * cov_matrix[2][2] + 2 * optimized_weights[0] * optimized_weights[2] * cov_matrix[0][2] + 2 * optimized_weights[0] * optimized_weights[1] * cov_matrix[0][1] + 2 * optimized_weights[1] * optimized_weights[2] * cov_matrix[1][2]) ** 0.5

    # Calculate the amounts to invest in each stock
    total_investment = 100000    # Total investment amount
    amount_x = optimized_weights[0] * total_investment
    amount_y = optimized_weights[1] * total_investment
    amount_z = optimized_weights[2] * total_investment

    print("Optimized Weights:", optimized_weights)
    print("Optimized Portfolio Standard Deviation:", optimized_portfolio_stddev)
    print(f"Amount to Invest in {n1}:", amount_x)
    print(f"Amount to Invest in {n2}:", amount_y)
    print(f"Amount to Invest in {n3}:", amount_z)
    print("Total Invested Amount:", amount_x + amount_y + amount_z)
    
    # Creating dataset
    stocks = [n1, n2, n3]
    data = [amount_x, amount_y , amount_z]

    # Creating explode data
    explode = (0.2, 0.1, 0.0)
    # Creating color parameters
    colors = ( "orange", "cyan", "brown", "grey", "indigo", "beige")
    # Wedge properties
    wp = { 'linewidth' : 1, 'edgecolor' : "green" }
    # Creating autocpt arguments
    def func(pct, allvalues):
        absolute = int(pct / 100.*no.sum(allvalues))
        return "{:.1f}%\n(₹{:d})".format(pct, absolute)

    # Creating plot
    fig, ax = plt.subplots(figsize =(10, 7))
    wedges, texts, autotexts = ax.pie(data, autopct = lambda pct: func(pct, data), explode = explode, labels = stocks, shadow = True, colors = colors, startangle = 90, wedgeprops = wp, textprops = dict(color ="magenta"))
    # Adding legend
    ax.legend(wedges, stocks, title ="Stocks", loc ="center left", bbox_to_anchor =(1, 0, 0.5, 1))
    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title(f"Optimized Portfolio \nTotal Invested Amount ₹{amount_x + amount_y + amount_z}")
    # show plot
    plt.show()

print("\nInitial Model Considering Data from Excel: Prices from 1st January 2022 to 30 December 2022")

# Mean returns
mean_return_R = -0.55422616099586
mean_return_T = 3.97536150640398
mean_return_L = 0.688590331343627

# Covariance matrix
cov_matrix = [
    [81.0742159483109, 14.3617343758859, 64.3425857361548],
    [14.3617343758859, 30.7585122170634, 19.6674773286696],
    [64.3425857361548, 19.6674773286696, 88.9136187064371]
]

# Negative of the objective function to minimize
def negative_sharpe_ratio(weights):
    portfolio_return = sum([mean_return_R * weights[0], mean_return_T * weights[1], mean_return_L * weights[2]])
    portfolio_stddev = (weights[0] ** 2 * cov_matrix[0][0] + weights[1] ** 2 * cov_matrix[1][1] + weights[2] ** 2 * cov_matrix[2][2] + 2 * weights[0] * weights[1] * cov_matrix[0][1] + 2 * weights[0] * weights[2] * cov_matrix[0][2] + 2 * weights[1] * weights[2] * cov_matrix[1][2]) ** 0.5
    sharpe_ratio = portfolio_return / portfolio_stddev
    return -sharpe_ratio

# Equality constraint (sum of weights == 1)
def constraint(weights):
    return sum(weights) - 1

# Initial guess for weights
initial_weights = [0.33, 0.33, 0.34]  # Equal initial allocation

# Bounds for weights
bounds = ((0, 1), (0, 1), (0, 1))  # Each weight between 0 and 1

# Equality constraint
constraints = {'type': 'eq', 'fun': constraint}

# Solve the optimization problem
result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Extract optimized weights
optimized_weights = result.x
optimized_portfolio_stddev = (optimized_weights[0] ** 2 * cov_matrix[0][0] + optimized_weights[1] ** 2 * cov_matrix[1][1] + optimized_weights[2] ** 2 * cov_matrix[2][2] + 2 * optimized_weights[0] * optimized_weights[2] * cov_matrix[0][2] + 2 * optimized_weights[0] * optimized_weights[1] * cov_matrix[0][1] + 2 * optimized_weights[1] * optimized_weights[2] * cov_matrix[1][2]) ** 0.5

# Calculate the amounts to invest in each stock
total_investment = 100000     # Total investment amount
amount_x = optimized_weights[0] * total_investment
amount_y = optimized_weights[1] * total_investment
amount_z = optimized_weights[2] * total_investment

print("Optimized Weights:", optimized_weights)
print("Optimized Portfolio Standard Deviation:", optimized_portfolio_stddev)
print("Amount to Invest in ASIANPAINT:", '{:.11f}'.format(amount_x))
print("Amount to Invest in ITC:", amount_y)
print("Amount to Invest in TITAN:", amount_z)
print("Total Invested Amount:", amount_x + amount_y + amount_z)

print("\nRevised Model Considering Data from Python: Prices from 24-09-2022 to 24-09-2023")
n1 = 'TCS'
n2 = 'TECHM'
n3 = 'HCLTECH'
# n3 = 'ONGC'
# n3 = 'COALINDIA'
model(n1, n2, n3)

# Thank You !!