import nltk

def main():

	#open the raw data file
	datafile = open('/tmp/data.txt')

	#remove all dots
	text = datafile.read().replace('.', ' ')

	#remove commas
	text = text.replace(',', ' ')

	#remove question marks
	text = text.replace('?', ' ')

	#remove exclamation
	text = text.replace('!', '')

	#remove double quotes
	text = text.replace('"', '')

	#remove single quotes
	text = text.replace('\'', '')

	#remove brackets
	text = text.replace('(', ' ')
	text = text.replace(')', ' ')

	#remove all slashes
	text = text.replace('\\', ' ')
	text = text.replace('/', ' ')

	#remove all hash
	text = text.replace('#', '')
	text = text.replace('-', '')

	text = text.replace('=', '')
	text = text.replace('_', '')
	text = text.replace(';', ' ')
	text = text.replace('&', '')
	text = text.replace(':', ' ')
	text = text.replace('`', '')


	outfile = open('/tmp/big_data.txt', 'w')
	outfile.write(text)
	outfile.close()


if __name__ == '__main__':
	main()
