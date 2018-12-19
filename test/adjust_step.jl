function adjust_step(tbar :: Float64, t :: Float64, dt :: Float64)

    lstat  = 0

    if (abs(tbar - (t + dt)) / tbar < 1.e-15)

        lstat = 1

    elseif (t + dt > tbar)

        dt    = tbar - t
        lstat = 1

    end

    dt, lstat

end
