local progress_bar = {}

progress_bar.__index = progress_bar

local function get_terminal_width()
    if os.getenv("TERM") then
        if os.execute("tput cols > /dev/null 2>&1") == 0 then
            return tonumber(io.popen("tput cols"):read("*a"))
        end
    end
    return 80
end

function progress_bar:new(total)
    local obj = {
        total = total,
        current = 0,
        start_time = os.clock(),
        bar_width = 40,
        term_width = get_terminal_width(),
    }
    setmetatable(obj, self)
    return obj
end

function progress_bar:clear_lines()
    for _ = 1, self.last_lines do
        io.write("\r\27[K")
        if _ < self.last_lines then
            io.write("\27[F")
        end
    end
end

function progress_bar:format_time(seconds)
    local hours = math.floor(seconds / 3600)
    local minutes = math.floor((seconds % 3600) / 60)
    local seconds = math.floor(seconds % 60)
    
    if hours > 0 then
        return string.format("%02d:%02d:%02d", hours, minutes, seconds)
    else
        return string.format("%02d:%02d", minutes, seconds)
    end
end

function progress_bar:display(str)
    local percentage = (self.current / self.total) * 100
    local rounded_percent = math.floor(percentage + 0.5)
    
    -- Calculate filled portion of the bar
    local filled = math.floor((self.current / self.total) * self.bar_width)
    local bar = string.rep("█", filled) .. string.rep(" ", self.bar_width - filled)
    
    -- Calculate time statistics
    local elapsed = os.clock() - self.start_time
    local elapsed_str = self:format_time(elapsed)
    
    local remaining = 0
    if self.current > 0 then
        remaining = (self.total - self.current) * (elapsed / self.current)
    end
    local remaining_str = self:format_time(remaining)
    
    -- Calculate iterations per second
    local it_per_second = 0
    if elapsed > 0 then
        it_per_second = self.current / elapsed
    end

    -- Create progress string
    local progress_str = string.format(
        "%3d%%|%s| %d/%d [%s<%s, %.2fit/s]",
        rounded_percent,
        bar,
        self.current,
        self.total,
        elapsed_str,
        remaining_str,
        it_per_second
    )

    -- Add padding to overwrite previous output
    progress_str = progress_str .. str

    -- Update the progress bar
    io.write("\r" .. progress_str)
    io.flush()
end

function progress_bar:step(keys)
    self.current = self.current + 1
    if self.current > self.total then
        self.current = self.total
    end

    if keys ~= nil then
        str = ""
        for k, v in pairs(keys) do
            str = str .. "[" .. k .. ": " .. v .. "] "
        end
    end
    self:display(str)
end

return progress_bar

-- Usage:
-- local total_steps = 10000
-- local bar = progress_bar:new(total_steps)

-- for i = 1, total_steps do
--     bar:step({learning_rate = 1.12, momentum = 0.9})
--     local t = os.clock()
--     while os.clock() - t < 0.001 do end
-- end