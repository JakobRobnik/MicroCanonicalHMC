salloc --nodes 1 --qos interactive --time 00:05:00 --image=reubenharry/cosmo:1.0 --constraint gpu --account m4031g
shifter --module=gpu --image=reubenharry/cosmo:1.0 python3 -m junk

