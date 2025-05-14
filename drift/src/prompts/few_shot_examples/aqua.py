aqua_examples = [
    {
        "question":"""
        The moon revolves around the earth at a speed of approximately 1.05 kilometers per second. 
        This approximate speed is how many kilometers per hour?
        """,
        "thought":"""
        There are 60 seconds in 1 minute and 60 minutes in 1 hour which equates to 3600 seconds in 1 hour.
        The calculation requires multiplying 1.05 kilometers per hour with 3600 seconds to get a final answer 
        in kilometers per hour. 
        """
    },
    {
        "question":"In what time will a train 120 meters long cross an electric pole, if its speed is 184 km/hr?",
        "thought":"""
        First convert speed into m/sec
        Speed = 184*(5/18) = 51 m/sec
        """
    },
    {
        "question":"A customer pays 30 dollars for a coffee maker after a discount of 20 dollars. What is the original price of the coffee maker?",
        "thought":"""
        Let x be the original price.
        x - 20 = 30
        x - 20 + 20 = 30 + 20
        """
    },
    {
        "question":"Evaluate: |7 - 8(3 - 12)| - |5 - 11| = ?",
        "thought":"""
        According to order of operations, inner brackets first. Hence
        |7 - 8(3 - 12)| - |5 - 11| = |7 - 8*(-9)| - |5 - 11|
        """
    },
    {
        "question":"A man can row his boat with the stream at 14 km/h and against the stream in 4 km/h. The man's rate is?",
        "thought":"""
        Assume v as the man's speed in stationary water.\n
        Assume s as the man's rate in the stream.\n
        Then, \n
        (v + s) is the speed with the stream. and (v - s) is the speed against the stream=. \n
        (v + s) - (v - s) = 14 - 4
        """
    },
    {
        "question":"A train leaves Delhi at 9 a.m. at a speed of 30 kmph. Another train leaves at 2 p.m. at a speed of 40 kmph on the same day and in the same direction. How far from Delhi, will the two trains meet?",
        "thought":"""
        Sine the first train leaves at 9 a.m. and the second train leaves at 2 p.m.. The second train leaves 5 hours after the first train.
        The amount of distance the first train covers in the first 5 hours is:
        30 kmph * 5 hours = 150 km
        The difference in the speed of the two trains is:
        40 kmph - 30 kmph = 10 kmph
        The next step will be to compute how long it will take train 2 to cover the distance train 1 has already covered.
        """
    },
    {
        "question":"If P is a prime number greater than 3, find the remainder when P^2 + 14 is divided by 12.",
        "thought":"""
        Every prime number greater than 3 can be written 6N+1 or 6N-1.
        If P = 6N+1, then P^2 + 14 = 36N^2 + 12N + 1 + 14 = 36N^2 + 12N + 12 + 3
        """
    },
    {
        "question":"A, B and C are partners in a business. Their capitals are respectively, Rs.5000, Rs.6000 and Rs.4000. A gets 30% of the total profit for managing the business. The remaining profit is divided among three in the ratio of their capitals. In the end of the year, the profit of A is Rs.200 more than the sum of the profits of B and C. Find the total profit.",
        "thought":"""
        A:B:C = 5000:6000:4000
        A:B:C = 5:6:4
        Let the total profit = 100 - 30 = 70
        5/15 * 70 = 70/3
        A share = 70/3 + 30 = 160/3
        """
    },
]