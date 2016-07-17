import numpy
import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')


def man_and_women_num():
    return "%d %d" % (data['Sex'].value_counts()['male'], data['Sex'].value_counts()['female'])


def people_survive():
    return "%0.2f" % ((data['Survived'].value_counts()[1] * 100) / float(data.count()[0]))


def people_pclass():
    return "%0.2f" % ((data['Pclass'].value_counts()[1] * 100) / float(data.count()[0]))


def people_age():
    return "%0.2f %0.2f" % (data['Age'].mean(), data['Age'].median())


def most_popular_name():
    data2 = data[data['Sex'] == 'female']
    data3 = pandas.DataFrame({'Name': data2['Name'].str.split('.').str[1]})
    result = data3['Name'].value_counts()

    return "%s".split(' ')[0] % (data3['Name'].value_counts())


def write_res(result, filename=""):
    with open(filename, 'w') as f:
        f.write(result)


if __name__ == '__main__':
    print(data.head())

    write_res(man_and_women_num(), filename='results/1.txt')

    write_res(people_survive(), filename='2.txt')

    write_res(people_pclass(), filename='3.txt')  # 24.24
    print("-------4")
    print(people_age())  # 28 29.7
    print("-------5")
    print(data.corr()['Parch']['SibSp'])  #
    print("-------6")
    print(most_popular_name())
