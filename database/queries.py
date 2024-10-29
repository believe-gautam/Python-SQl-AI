# database/queries.py
class Queries:
    # Daily Performance Metrics
    DAILY_PERFORMANCE = """
    SELECT 
        `Date`,
        `Overall Sales (£)`,
        `Overall Average Prep Time (Min)`,
        `Overall number of order (Orders)`,
        `Overall average rider waiting time (min)`,
        `Overall average open rate (%)`,
        `Overall rejected orders (%)`,
        `Overall cancelled orders (%)`
    FROM daily_output
    WHERE Date BETWEEN %s AND %s
    ORDER BY Date
    """
    
    # Platform Comparison
    PLATFORM_COMPARISON = """
    SELECT 
        `Date`,
        `Deliveroo Sales (£)`,
        `Uber Sales (£)`,
        `JustEat Sales (£)`,
        `Deliveroo - number of order (Orders)`,
        `Uber - number of order (Orders)`,
        `JustEat - number of order (Orders)`
    FROM daily_output
    WHERE Date BETWEEN %s AND %s
    ORDER BY Date
    """
    
    # Weekly Performance Metrics
    WEEKLY_METRICS = """
    SELECT 
        `Week Start`,
        `Week End`,
        `Overall Sales (£)`,
        `Overall Prep Time (Min)`,
        `Overall number of orders (Orders)`,
        `Overall rider waiting time (min)`,
        `Overall open rate (%)`,
        `Overall rejected orders (%)`,
        `Overall cancelled orders (%)`,
        `Overall average rating`,
        `Overall average overallscore`
    FROM weekly_output
    WHERE `Week Start` BETWEEN %s AND %s
    ORDER BY `Week Start`
    """
    
    # Quality Metrics
    QUALITY_METRICS = """
    SELECT 
        `Week Start`,
        `Total missing orders`,
        `Order never received`,
        `Partial items in the order`,
        `Food Quality issue with order`,
        `Total number of incorrect items`,
        `Total number of items prepared incorrectly`,
        `Overall average rating`
    FROM weekly_output
    WHERE `Week Start` BETWEEN %s AND %s
    ORDER BY `Week Start`
    """