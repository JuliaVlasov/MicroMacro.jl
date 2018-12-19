function adjust_step(tbar, t, dt)

    lstat = 0

    if (abs(tbar - (t + dt)) / tbar < 1.e-15)

        lstat = 1

    elseif (t + dt > tbar)

        dt    = tbar - t
        lstat = 1

    end

    dt, lstat

end
