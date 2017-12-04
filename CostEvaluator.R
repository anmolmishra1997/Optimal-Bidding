d = 300;
battery = 0;
Bid_Price <- data.frame(matrix(6.5, ncol = 24, nrow = d));
#Bid_Price <- Price_Tst_pred;
Bid_Quantity <- data.frame(matrix(0, ncol = 24, nrow = d));
for (i in 1:d) {
  for (j in 1:24) {
  #  Bid_Quantity[i, j] = max (Demand_Tst_pred[i, j] - Solar_Tst_pred[i, j], 0);
  }
}

Cost <- data.frame(matrix(0, ncol = 24, nrow = d));
for (i in 1:d) {
  for(j in 1:24) {
    quantity = 0;
    quantity = quantity + Solar_Tst[i, j];
    if (quantity > Demand_Tst[i,j])
      Bid_Quantity[i,j] = 0;
    
    if (Bid_Price[i,j] >= Price_Tst[i,j])
    {
      quantity = quantity + Bid_Quantity[i, j];
      Cost[i, j] = Cost[i, j] + Price_Tst[i, j] * Bid_Quantity[i, j];
    }
    
    if (quantity > Demand_Tst[i, j])
    {
      battery = min (min (quantity - Demand_Tst[i, j], 5) + battery, 25);
    }
    else if (quantity < Demand_Tst[i, j])
    {
      temp = quantity;
      quantity = min(Demand_Tst[i, j], quantity + 4 * min(battery, 5)/5);
      battery  = battery - (quantity-temp);
    }
    
    if (quantity < Demand_Tst[i, j])
    {
      Cost[i, j] = Cost[i, j] + (Demand_Tst[i, j] - quantity)*7;
    }
  }
}