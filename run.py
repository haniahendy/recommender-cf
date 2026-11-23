from rec.cf import leave_one_out_hit_rate, make_ratings

R, _, _ = make_ratings()
print(leave_one_out_hit_rate(R, k=5))
