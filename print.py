import main

main.EPOCHS = 15

# train with these parameters
# record sq_loss and err at every half epoch
# keep going until 15 epochs

main.PICKLE = "nn10-50.pickle"
main.HIDDEN_UNITS = 10
main.MINI_BATCH = 50
main.main()

main.PICKLE = "nn10-100.pickle"
main.HIDDEN_UNITS = 10
main.MINI_BATCH = 100
main.main()

main.PICKLE = "nn10-500.pickle"
main.HIDDEN_UNITS = 10
main.MINI_BATCH = 500
main.main()


main.PICKLE = "nn50-50.pickle"
main.HIDDEN_UNITS = 50
main.MINI_BATCH = 50
main.main()

main.PICKLE = "nn50-100.pickle"
main.HIDDEN_UNITS = 50
main.MINI_BATCH = 100
main.main()

main.PICKLE = "nn50-500.pickle"
main.HIDDEN_UNITS = 50
main.MINI_BATCH = 500
main.main()


main.PICKLE = "nn100-50.pickle"
main.HIDDEN_UNITS = 100
main.MINI_BATCH = 50
main.main()

main.PICKLE = "nn100-100.pickle"
main.HIDDEN_UNITS = 100
main.MINI_BATCH = 100
main.main()

main.PICKLE = "nn100-500.pickle"
main.HIDDEN_UNITS = 100
main.MINI_BATCH = 500
main.main()


main.PICKLE = "nn500-50.pickle"
main.HIDDEN_UNITS = 500
main.MINI_BATCH = 50
main.main()

main.PICKLE = "nn500-100.pickle"
main.HIDDEN_UNITS = 500
main.MINI_BATCH = 100
main.main()

main.PICKLE = "nn500-500.pickle"
main.HIDDEN_UNITS = 500
main.MINI_BATCH = 500
main.main()


main.PICKLE = "nn1000-50.pickle"
main.HIDDEN_UNITS = 1000
main.MINI_BATCH = 50
main.main()

main.PICKLE = "nn1000-100.pickle"
main.HIDDEN_UNITS = 1000
main.MINI_BATCH = 100
main.main()

main.PICKLE = "nn1000-500.pickle"
main.HIDDEN_UNITS = 1000
main.MINI_BATCH = 500
main.main()
