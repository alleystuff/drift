mathqa_examples = [
    {
        "question": "the ratio of the radius of two circles is 2 : 3, and then the ratio of their areas is?",
        "thought": """
        Let radius of circle 1 be: r1
        Let radius of circle 2 be: r2
        The ratio is:
        r1:r2
        2:3

        The formula for area of a circle is: πr^2
        Area of a circle uses squared radius.
        """
    },
    {
        "question":"if x = 1 - 5 t and y = 2 t - 1 , then for what value of t does x = y?",
        "thought":"""
        we are given x = 1 - 5 t and y = 2 t - 1 , and we need to determine the value for t when x = y . we should notice that both x and y are already in terms of t . thus , we can substitute 1 – 5 t for x and 2 t – 1 for y in the equation x = y .
        """
    },
    {
        "question":"the sum of three consecutive even numbers is 87 . find the middle number of the three ?",
        "thought":"""
        Since the three numbers are consecutive. Divide the sum by 3.
        """
    },
    {
        "question":"60\%\ of a number is added to 160 , the result is the same number . find the number ?",
        "thought":"""
        Set up an equation. Assume the number is x, then 60\%\ of x is 0.6*x.
        Add 0.6*x to 160 which should be equal to x. Then find x. 
        """
    },
    {
        "question":"the ratio of two numbers is 3 : 4 and their sum is 21 . the greater of the two numbers is ?",
        "thought":"""
        Let the two numbers be 3x and 4x, where x is a common multiplier. Then find x.
        """
    },
    {
        "question":"a , b , and c are integers and a < b < c . s is the set of all integers from a to b , inclusive . q is the set of all integers from b to c , inclusive . the median of set s is ( 3 / 4 ) * b . the median of set q is ( 7.5 / 8 ) * c . if r is the set of all integers from a to c , inclusive , what fraction of c is the median of set r ?",
        "thought":"""
        For a consecutive set of numbers the median is equal to mean. So for example for all numbers
        x + (x+1) + (x+y) + ... + y the mean is (x+y)/2 which is also the median.
        """
    },
    {
        "question":"the sum of ages of 6 children born at the intervals of 3 years each is 75 years . what is the age of the youngest child ?",
        "thought":"""
        let the ages of children be x , ( x + 3 ) , ( x + 6 ) , ( x + 9 ) , ( x + 12 ) , ( x + 15 ) years . then , x + ( x + 3 ) + ( x + 6 ) + ( x + 9 ) + ( x + 12 ) + ( x + 15 ) = 75
        """
    },
    {
        "question":"rates for having a manuscript typed at a certain typing service are $ 6 per page for the first time a page is typed and $ 3 per page each time a page is revised . if a certain manuscript has 100 pages , of which 30 were revised only once , 10 were revised twice , and the rest required no revisions , what was the total cost of having the manuscript typed ?",
        "thought":"""
        for 100 - 30 - 10 = 60 pages only cost is 6 $ per page for the first time page is typed - 60 * 5 = 300 $ 
        """
    }
]